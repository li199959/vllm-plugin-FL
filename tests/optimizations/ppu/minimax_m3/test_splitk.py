import pytest
import torch

from vllm_fl.optimizations.ppu.minimax_m3.splitk import splitk_linear, splitk_linear_impl

pytestmark = pytest.mark.gpu


def _reference(x, w, b):
    return torch.nn.functional.linear(x.double(), w.double(), None if b is None else b.double())


@pytest.mark.parametrize("dtype,n", [(torch.float32, 128), (torch.bfloat16, 1024), (torch.bfloat16, 384)])
@pytest.mark.parametrize("m", [1, 7, 16, 33, 64, 100, 512])
@pytest.mark.parametrize("with_bias", [False, True])
@torch.inference_mode()
def test_splitk_matches_reference(dtype, n, m, with_bias):
    torch.manual_seed(m)
    k = 6144
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    w = torch.randn(n, k, device="cuda", dtype=dtype) * 0.02
    b = torch.randn(n, device="cuda", dtype=dtype) if with_bias else None
    ref = _reference(x, w, b)
    got = splitk_linear_impl(x, w, b).double()
    native = torch.nn.functional.linear(x, w, b).double()
    # Must be no worse than the native GEMM (plus one output ulp for bf16).
    slack = 1e-6 if dtype == torch.float32 else 2 ** -7 * ref.abs().max().item()
    assert (got - ref).abs().max() <= (native - ref).abs().max() + slack
    if dtype == torch.float32:
        assert (got - ref).abs().max() < 1e-5


@pytest.mark.parametrize("m", [1, 32, 300])
@torch.inference_mode()
def test_splitk_graph_and_determinism(m):
    x = torch.randn(m, 6144, device="cuda")
    w = torch.randn(128, 6144, device="cuda") * 0.02
    first = splitk_linear(x, w, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = splitk_linear(x, w, None)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, first)


@torch.inference_mode()
def test_router_routing(monkeypatch):
    """fp32 router linears at graph sizes go to split-K, larger ones stay native."""
    from vllm_fl.optimizations.ppu.minimax_m3 import linear

    calls = []
    monkeypatch.setattr(linear, "splitk_linear", lambda x, w, b: calls.append(x.shape[0]) or x @ w.T)
    monkeypatch.setattr(linear, "_original_linear_apply", lambda self, layer, x, b=None: "native")

    class Layer:
        weight = torch.randn(128, 6144, device="cuda")

    for m in (1, 64, 512):
        linear._hybrid_linear_apply(None, Layer, torch.randn(m, 6144, device="cuda"))
    assert calls == [1, 64, 512]
    assert linear._hybrid_linear_apply(None, Layer, torch.randn(513, 6144, device="cuda")) == "native"
