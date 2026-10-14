"""Keep Gems kernels from recompiling on every serving batch size.

Gems bakes sizes that only bound masks into the compiled kernel: the
linearized ``index`` generator writes the indexed dimension into the source,
and ``moe_align_block_size_stage1`` declares output lengths ``constexpr``.
Chunked prefill produces a new token count almost every step, so each one
triggered a Triton compile and stalled all TP ranks for seconds. These
variants pass those sizes at runtime; kernel bodies and configs are unchanged.
"""
import importlib.util
import os

import triton
import triton.language as tl

_installed = False


@triton.jit(do_not_specialize=[
    "numel", "tokens_per_thread", "numel_sorted_token_ids", "numel_expert_ids",
    "block_size_sorted", "block_size_expert",
])
def _moe_align_stage1_kernel(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids,
    numel_expert_ids,
    block_size_sorted,
    block_size_expert,
    TILE: tl.constexpr,
):
    # Gems stage1 with each program's span walked in fixed tiles: the spans
    # (and therefore which program touches which element) are unchanged, but
    # they are runtime values instead of power-of-two constexprs.
    pid = tl.program_id(0)
    lanes = tl.arange(0, TILE)

    for base in range(0, block_size_sorted, TILE):
        local = base + lanes
        offsets_sorted = pid * block_size_sorted + local
        mask_sorted = (local < block_size_sorted) & (offsets_sorted < numel_sorted_token_ids)
        tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    for base in range(0, block_size_expert, TILE):
        local = base + lanes
        offsets_expert = pid * block_size_expert + local
        mask_expert = (local < block_size_expert) & (offsets_expert < numel_expert_ids)
        tl.store(expert_ids_ptr + offsets_expert, -1, mask=mask_expert)

    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts
    for base in range(0, tokens_per_thread, TILE):
        local = base + lanes
        offsets = start_idx + local
        mask = (local < tokens_per_thread) & (offsets < numel)
        expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
        tl.atomic_add(tokens_cnts_ptr + off_c + expert_id, 1, mask=mask)


class _Stage1Launcher:
    """Accepts Gems' ``stage1[grid](...)`` call and launches the tiled kernel."""

    fn = _moe_align_stage1_kernel

    def __getitem__(self, grid):
        def launch(*args):
            return _moe_align_stage1_kernel[grid](*args, TILE=256)
        return launch


moe_align_block_size_stage1 = _Stage1Launcher()


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _generate_single_index(prefix_size, suffix_size, configs, code):
    """Gems' linearized kernel for one index tensor, indexed size at runtime."""
    from flag_gems.ops.index import _write_index_configs, generate_imports

    code = generate_imports(code)
    code.writeline("@libentry()")
    code.writeline("@libtuner(")
    with code.indent():
        _write_index_configs(code, configs)
        code.writeline('key=["M"],')
        code.writeline('strategy=["align32"],')
        code.writeline("warmup=5,")
        code.writeline("rep=10,")
    code.writeline(")")
    code.writeline("@triton.jit")
    code.writeline("def _index_linearized_jit_function(")
    with code.indent():
        code.writelines([
            "input_ptr,", "indices0_ptr,", "out_ptr,", "M,", "N,", "D0,",
            "BLOCK_SIZE0: tl.constexpr,", "BLOCK_SIZE1: tl.constexpr,",
        ])
    code.writeline("):")
    with code.indent():
        code.writelines([
            "pid_p = tl.program_id(axis=0)",
            "pid_m = tl.program_id(axis=1)",
            "pid_n = tl.program_id(axis=2)",
            "index_offsets = pid_m * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)",
            "suffix_offsets = pid_n * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)",
            "index_mask = index_offsets < M",
            "raw0 = tl.load(indices0_ptr + index_offsets, mask=index_mask, other=0)",
            "idx0 = tl.where(raw0 < 0, raw0 + D0, raw0)",
            "valid_index = (idx0 >= 0) & (idx0 < D0)",
            f"src_offsets = (pid_p * D0 + idx0[:, None]) * {suffix_size} + suffix_offsets[None, :]",
            f"out_offsets = (pid_p * M + index_offsets[:, None]) * {suffix_size} + suffix_offsets[None, :]",
            "mask = index_mask[:, None] & valid_index[:, None] & "
            f"(suffix_offsets[None, :] < {suffix_size})",
            "cur_value = tl.load(input_ptr + src_offsets, mask=mask)",
            "tl.store(out_ptr + out_offsets, cur_value, mask=mask)",
        ])
    code.newline()
    code.writeline("def _index_linearized_wrapper(input, start_dim, indices, out):")
    with code.indent():
        code.writelines([
            "M = indices[0].numel()",
            f"N = {suffix_size}",
            "grid = lambda meta: (",
            f"    {prefix_size},",
            "    triton.cdiv(M, meta['BLOCK_SIZE0']),",
            "    triton.cdiv(N, meta['BLOCK_SIZE1']),",
            ")",
            "_index_linearized_jit_function[grid](",
            "    input, indices[0], out, M, N, input.shape[start_dim],",
            ")",
            "return input",
        ])
    return code


class ShapeStableIndexFunction:
    """Drop-in for Gems' LinearizedAdjacentIndexFunction.

    Single-index calls are keyed on what the generated source depends on
    (prefix/suffix volume, dtype and the chosen configs), not the indexed
    size. Multi-index calls keep the original Gems implementation.
    """

    def __init__(self, fallback):
        self.fallback = fallback
        self.overloads = {}

    def __call__(self, inp, start_dim, tensor_indices, out):
        if len(tensor_indices) != 1:
            return self.fallback(inp, start_dim, tensor_indices, out)
        from flag_gems.ops.index import _index_candidate_configs
        from flag_gems.utils.code_cache import code_cache_dir
        from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

        start_dim = int(start_dim)
        prefix_size = _volume(inp.shape[:start_dim])
        suffix_size = _volume(inp.shape[start_dim + 1:])
        configs = _index_candidate_configs(inp, start_dim, 1, tensor_indices)
        config_key = tuple(
            (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages) for c in configs
        )
        key = (prefix_size, suffix_size, inp.dtype, config_key)
        overload = self.overloads.get(key)
        if overload is None:
            code = _generate_single_index(prefix_size, suffix_size, configs, IndentedBuffer())
            name = (f"index_linearized_fl_p{prefix_size}_n{suffix_size}_"
                    f"{str(inp.dtype).replace('.', '_')}_{abs(hash(config_key)):x}_{os.getpid()}")
            path = code_cache_dir() / f"{name}.py"
            write_atomic(path, code.getvalue())
            spec = importlib.util.spec_from_file_location(f"_gen_module_{name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            overload = self.overloads[key] = module._index_linearized_wrapper
        return overload(inp, start_dim, tensor_indices, out)


def install_shape_stable_gems():
    global _installed
    if _installed:
        return
    import sys

    import flag_gems  # noqa: F401  (loads the submodules below)

    # flag_gems.ops.index is shadowed by the exported function; use sys.modules.
    moe_align = sys.modules["flag_gems.fused.moe_align_block_size"]
    gems_index = sys.modules["flag_gems.ops.index"]
    moe_align.moe_align_block_size_stage1 = moe_align_block_size_stage1
    gems_index._linearized_adjacent_index_func = ShapeStableIndexFunction(
        gems_index._linearized_adjacent_index_func
    )
    _installed = True
