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


@triton.jit(
    do_not_specialize=[
        "numel",
        "tokens_per_thread",
        "numel_sorted_token_ids",
        "numel_expert_ids",
        "block_size_sorted",
        "block_size_expert",
    ]
)
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
        mask_sorted = (local < block_size_sorted) & (
            offsets_sorted < numel_sorted_token_ids
        )
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
        code.writelines(
            [
                "input_ptr,",
                "indices0_ptr,",
                "out_ptr,",
                "M,",
                "N,",
                "D0,",
                "BLOCK_SIZE0: tl.constexpr,",
                "BLOCK_SIZE1: tl.constexpr,",
            ]
        )
    code.writeline("):")
    with code.indent():
        code.writelines(
            [
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
            ]
        )
    code.newline()
    code.writeline("def _index_linearized_wrapper(input, start_dim, indices, out):")
    with code.indent():
        code.writelines(
            [
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
            ]
        )
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
        suffix_size = _volume(inp.shape[start_dim + 1 :])
        configs = _index_candidate_configs(inp, start_dim, 1, tensor_indices)
        config_key = tuple(
            (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages)
            for c in configs
        )
        key = (prefix_size, suffix_size, inp.dtype, config_key)
        overload = self.overloads.get(key)
        if overload is None:
            code = _generate_single_index(
                prefix_size, suffix_size, configs, IndentedBuffer()
            )
            name = (
                f"index_linearized_fl_p{prefix_size}_n{suffix_size}_"
                f"{str(inp.dtype).replace('.', '_')}_{abs(hash(config_key)):x}_{os.getpid()}"
            )
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


# --- copy_ -------------------------------------------------------------------
# Gems' pointwise_dynamic declares every stride ``tl.constexpr``. copy_ between a
# padded persistent buffer and a fresh slice (M-RoPE positions [3, N] with row
# stride max_tokens+1 vs N; logits rearranged per batch size) therefore compiles
# a new kernel for almost every token count. The kernel below takes sizes and
# strides at runtime; fully contiguous copies keep using Gems' rank-1 kernel.

_copy_installed = False
_MAX_RANK = 4


@triton.jit(
    do_not_specialize=[
        "numel",
        "s1",
        "s2",
        "s3",
        "ss0",
        "ss1",
        "ss2",
        "ss3",
        "ds0",
        "ds1",
        "ds2",
        "ds3",
    ]
)
def _strided_copy_kernel(
    src_ptr,
    dst_ptr,
    numel,
    s1,
    s2,
    s3,
    ss0,
    ss1,
    ss2,
    ss3,
    ds0,
    ds1,
    ds2,
    ds3,
    BLOCK: tl.constexpr,
):
    # Shapes are left-padded to rank 4; dim 0's size is implied by numel.
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    i3 = offs % s3
    rest = offs // s3
    i2 = rest % s2
    rest = rest // s2
    i1 = rest % s1
    i0 = rest // s1
    src = i0 * ss0 + i1 * ss1 + i2 * ss2 + i3 * ss3
    dst = i0 * ds0 + i1 * ds1 + i2 * ds2 + i3 * ds3
    x = tl.load(src_ptr + src, mask=mask)
    tl.store(dst_ptr + dst, x.to(dst_ptr.dtype.element_ty), mask=mask)


def _padded(values, fill):
    return [fill] * (_MAX_RANK - len(values)) + list(values)


def _launch_strided_copy(src, dst):
    numel = dst.numel()
    sizes = _padded(dst.shape, 1)
    ss = _padded(src.stride(), 0)
    ds = _padded(dst.stride(), 0)
    block = 1024
    _strided_copy_kernel[(triton.cdiv(numel, block),)](
        src,
        dst,
        numel,
        sizes[1],
        sizes[2],
        sizes[3],
        *ss,
        *ds,
        BLOCK=block,
        num_warps=4,
    )
    return dst


class ShapeStableCopyKernel:
    """Stands in for Gems' ``_copy_kernel`` inside ``flag_gems.ops.copy``.

    ``copy_`` validates, broadcasts ``src`` to ``dst.shape`` and then calls
    ``_copy_kernel.instantiate(ndim)(src, out0=dst)``; only that last step
    changes. Rank 0 and rank > 4 keep the Gems overload.
    """

    def __init__(self, gems_kernel):
        self.gems = gems_kernel

    def instantiate(self, ndim):
        return self._copy

    def _copy(self, src, out0):
        dst = out0
        ndim = dst.ndim
        if ndim == 0 or ndim > _MAX_RANK:
            return self.gems.instantiate(ndim)(src, out0=dst)
        if src.is_contiguous() and dst.is_contiguous():
            # Rank-1 strides are always 1, so this overload never respecializes.
            self.gems.instantiate(1)(src.view(-1), out0=dst.view(-1))
            return dst
        return _launch_strided_copy(src, dst)


def install_shape_stable_copy():
    global _copy_installed
    if _copy_installed:
        return
    import sys

    import flag_gems  # noqa: F401

    gems_copy = sys.modules["flag_gems.ops.copy"]
    gems_copy._copy_kernel = ShapeStableCopyKernel(gems_copy._copy_kernel)
    _copy_installed = True
