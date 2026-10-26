import types

import pytest
import torch

from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune
from vllm_fl.optimizations.ppu.bge_m3 import linear

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _setup():
    _disable_flaggems_runtime_autotune()


def _weights(name):
    n, k = linear.SHAPES[name]
    w = (torch.randn(n, k, device="cuda") / k**0.5).half()
    b = torch.randn(n, device="cuda").half()
    return w, b


@pytest.mark.parametrize("name", sorted(linear.SHAPES))
@pytest.mark.parametrize("m", [1, 17, 512, 1000, 2048, 8192])
@torch.inference_mode()
def test_gems_linear_matches_native(name, m):
    torch.manual_seed(m)
    w, b = _weights(name)
    x = torch.randn(m, w.shape[1], device="cuda").half()
    ref = torch.nn.functional.linear(x.float(), w.float(), b.float())
    out = linear.gems_linear(x, w, b)
    assert out.shape == (m, w.shape[0]) and out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)


@torch.inference_mode()
def test_empty_batch():
    w, b = _weights("o")
    out = linear.gems_linear(torch.empty(0, 1024, device="cuda").half(), w, b)
    assert out.shape == (0, 1024)


@torch.inference_mode()
def test_apply_routes_only_selected_projections(monkeypatch):
    seen = []
    monkeypatch.setattr(linear, "_selected", frozenset({"o"}))
    monkeypatch.setattr(linear, "_original_linear_apply",
                        lambda self, layer, x, bias=None: seen.append(layer.weight.shape)
                        or torch.nn.functional.linear(x, layer.weight, bias))
    o = types.SimpleNamespace(weight=_weights("o")[0])
    fc1 = types.SimpleNamespace(weight=_weights("fc1")[0])
    x = torch.randn(64, 1024, device="cuda").half()
    linear._apply(None, o, x)
    linear._apply(None, fc1, x)
    linear._apply(None, o, x.t().contiguous().t())  # column-major input falls back
    assert seen == [torch.Size([4096, 1024]), torch.Size([1024, 1024])]


@torch.inference_mode()
def test_small_m_split(monkeypatch):
    calls = []
    real = linear._gems_linear
    monkeypatch.setattr(linear, "_gems_linear", lambda *a: calls.append(a[0].shape[0]) or real(*a))
    monkeypatch.setattr(linear, "_selected", frozenset({"o"}))
    monkeypatch.setattr(linear, "_small_m", 512)
    w, b = _weights("fc1")
    fc1 = types.SimpleNamespace(weight=w)
    for m in (256, 512, 513, 4096):
        x = torch.randn(m, 1024, device="cuda").half()
        assert linear.projection(x, w) == "fc1"
        out = linear._apply(None, fc1, x, b)
        torch.testing.assert_close(out, torch.nn.functional.linear(x, w, b), atol=2e-2, rtol=1e-2)
    assert calls == [256, 512]  # larger fc1 batches stay on the fallback (cuBLAS) path
    monkeypatch.setattr(linear, "_small_m", 0)
    assert linear.projection(x, w) is None


@torch.inference_mode()
def test_cuda_graph_replay():
    w, b = _weights("o")
    x = torch.randn(512, 1024, device="cuda").half()
    out = linear.gems_linear(x, w, b)  # compile outside capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = linear.gems_linear(x, w, b)
    x.mul_(0.5)
    g.replay()
    torch.testing.assert_close(out, torch.nn.functional.linear(x, w, b), atol=2e-2, rtol=1e-2)
