import pytest
import torch

from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune
from vllm_fl.optimizations.ppu.minimax_m3.linear import bf16_linear

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("rows", [1, 64, 129, 8192])
@pytest.mark.parametrize("k,n", [(6144, 1024), (6144, 384), (512, 6144), (192, 6144)])
@torch.inference_mode()
def test_hybrid_bf16_linear(rows, k, n):
    _disable_flaggems_runtime_autotune()
    torch.manual_seed(17)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(x, w, bias)
    torch.testing.assert_close(bf16_linear(x, w, bias), reference, rtol=0.03, atol=0.05)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = bf16_linear(x, w, bias)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.05)


@pytest.mark.parametrize("rows,cols", [(1024, 1), (1024, 32), (64, 200064)])
@torch.inference_mode()
def test_gems_index_graph(rows, cols):
    import flag_gems

    _disable_flaggems_runtime_autotune()
    x = torch.randn(rows, cols, device="cuda")
    ids = torch.tensor([0, 31, 1, -1], device="cuda", dtype=torch.int64)
    expected = x[ids]
    torch.testing.assert_close(flag_gems.index(x, [ids]), expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flag_gems.index(x, [ids])
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("rows", [7, 26, 32])
@torch.inference_mode()
def test_generated_index_does_not_benchmark(rows, monkeypatch):
    import flag_gems
    from flag_gems.utils.libentry import LibTuner

    _disable_flaggems_runtime_autotune()

    def fail_benchmark(*args, **kwargs):
        raise AssertionError("Serving must not benchmark first-seen index shapes")

    monkeypatch.setattr(LibTuner, "_bench", fail_benchmark)
    x = torch.arange(rows, device="cuda", dtype=torch.int32)
    ids = torch.tensor([0, -1, 1], device="cuda", dtype=torch.int64)
    torch.testing.assert_close(flag_gems.index(x, [ids]), x[ids])
