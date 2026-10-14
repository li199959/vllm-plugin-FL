import pytest
import torch

from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune
from vllm_fl.optimizations.ppu.minimax_m3.shape_stable import install_shape_stable_gems

pytestmark = pytest.mark.gpu


def _compiled(jit_fn):
    while not hasattr(jit_fn, "device_caches"):
        jit_fn = jit_fn.fn
    return sum(len(cache[0]) for cache in jit_fn.device_caches.values())


@pytest.fixture(autouse=True)
def _setup():
    _disable_flaggems_runtime_autotune()
    install_shape_stable_gems()


@pytest.mark.parametrize("cols,dtype", [(6144, torch.bfloat16), (1, torch.int32), (32, torch.float32)])
@torch.inference_mode()
def test_index_matches_torch_across_sizes(cols, dtype):
    import flag_gems

    torch.manual_seed(3)
    for rows in (3, 17, 64, 1135, 4155, 5831):
        x = torch.randn(rows, cols, device="cuda").to(dtype)
        if cols == 1:
            x = x.view(rows)
        ids = torch.randint(-rows, rows, (min(rows, 40),), device="cuda")
        torch.testing.assert_close(flag_gems.index(x, [ids]), x[ids], rtol=0, atol=0)


@torch.inference_mode()
def test_index_prefix_dim_and_graph():
    import flag_gems

    x = torch.randn(3, 97, 8, device="cuda")
    ids = torch.tensor([0, -1, 50, 96], device="cuda")
    torch.testing.assert_close(flag_gems.index(x, [None, ids]), x[:, ids], rtol=0, atol=0)
    pair = [torch.tensor([0, 2], device="cuda"), torch.tensor([1, -1], device="cuda")]
    torch.testing.assert_close(flag_gems.index(x, pair), x[pair[0], pair[1]], rtol=0, atol=0)
    expected = x[:, ids]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flag_gems.index(x, [None, ids])
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_index_does_not_recompile_per_size():
    import flag_gems
    import sys

    gems_index = sys.modules["flag_gems.ops.index"]

    func = gems_index._linearized_adjacent_index_func
    ids = torch.tensor([0, 1, -1], device="cuda")
    # Overloads are process-global; count only those this loop adds.
    existing = set(func.overloads)
    for rows in range(40, 100):
        x = torch.arange(rows, device="cuda", dtype=torch.int32)
        torch.testing.assert_close(flag_gems.index(x, [ids]), x[ids])
    added = [func.overloads[k] for k in set(func.overloads) - existing]
    assert len(added) <= 1
    key = next(k for k in func.overloads if k[:3] == (1, 1, torch.int32))
    kernel = func.overloads[key].__globals__["_index_linearized_jit_function"]
    # Triton may still split on divisibility of the runtime size, but bounded.
    assert _compiled(kernel) <= 4


def _reference_align(topk_ids, block_size, num_experts):
    flat = topk_ids.flatten().tolist()
    sorted_ids, expert_ids = [], []
    for expert in range(num_experts):
        tokens = [i for i, e in enumerate(flat) if e == expert]
        padded = -(-len(tokens) // block_size) * block_size
        sorted_ids += tokens + [len(flat)] * (padded - len(tokens))
        expert_ids += [expert] * (padded // block_size)
    return sorted_ids, expert_ids


@pytest.mark.parametrize("block_size", [16, 64])
@torch.inference_mode()
def test_moe_align_matches_reference_without_recompiles(block_size):
    import sys

    moe_align = sys.modules["flag_gems.fused.moe_align_block_size"]

    torch.manual_seed(5)
    before = None
    for tokens in (129, 200, 513, 1135, 2047, 4096, 5831):
        topk = torch.stack([torch.randperm(128, device="cuda")[:4] for _ in range(tokens)])
        topk = topk.to(torch.int32)
        sorted_ids, expert_ids, total = moe_align.moe_align_block_size(topk, block_size, 128)
        ref_sorted, ref_experts = _reference_align(topk.cpu(), block_size, 128)
        n = int(total.item())
        assert n == len(ref_sorted)
        experts = expert_ids[: n // block_size].tolist()
        assert experts == ref_experts
        # Token order inside one expert's segment depends on atomics.
        got, want = sorted_ids[:n].cpu(), torch.tensor(ref_sorted, dtype=torch.int32)
        per_row = torch.tensor(experts).repeat_interleave(block_size)
        for expert in per_row.unique():
            segment = per_row == expert
            assert torch.equal(got[segment].sort().values, want[segment].sort().values)
        assert (sorted_ids[n:] == topk.numel()).all()
        if before is None:
            before = _compiled(moe_align.moe_align_block_size_stage1)
    assert _compiled(moe_align.moe_align_block_size_stage1) == before
