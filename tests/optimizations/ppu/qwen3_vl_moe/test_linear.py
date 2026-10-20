import types

import pytest
import torch

from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune
from vllm_fl.optimizations.ppu.qwen3_vl_moe import linear

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _setup():
    _disable_flaggems_runtime_autotune()


@pytest.mark.parametrize("n", [128, 2048, 2560])
@pytest.mark.parametrize("m", [1, 7, 16, 33, 64, 100, 128])
@torch.inference_mode()
def test_tuned_linear_matches_native(m, n):
    torch.manual_seed(m * n)
    x = torch.randn(m, 2048, device="cuda").bfloat16()
    w = (torch.randn(n, 2048, device="cuda") * 0.02).bfloat16()
    ref = (x.float() @ w.float().t())
    out = linear.bf16_linear(x, w, None)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1.6e-2)


@torch.inference_mode()
def test_out_of_scope_shapes_fall_back(monkeypatch):
    calls = []
    real = torch.nn.functional.linear
    monkeypatch.setattr(torch.nn.functional, "linear",
                        lambda *a: calls.append(1) or real(*a))
    x = torch.randn(129, 2048, device="cuda").bfloat16()       # prefill size
    w = (torch.randn(2560, 2048, device="cuda") * 0.02).bfloat16()
    linear.bf16_linear(x, w, None)
    x2 = torch.randn(8, 1152, device="cuda").bfloat16()       # vision tower width
    w2 = (torch.randn(2048, 1152, device="cuda") * 0.02).bfloat16()
    linear.bf16_linear(x2, w2, None)
    assert calls == [1, 1]


@torch.inference_mode()
def test_apply_routes_only_text_shapes_and_graph_replay(monkeypatch):
    seen = []
    monkeypatch.setattr(linear, "_original_linear_apply",
                        lambda self, layer, x, bias=None: seen.append(x.shape) or x @ layer.weight.t())
    layer = types.SimpleNamespace(weight=(torch.randn(2560, 2048, device="cuda") * 0.02).bfloat16())
    other = types.SimpleNamespace(weight=(torch.randn(4096, 2048, device="cuda") * 0.02).bfloat16())
    x = torch.randn(64, 2048, device="cuda").bfloat16()
    out = linear._apply(None, layer, x)
    linear._apply(None, other, x)
    assert seen == [torch.Size([64, 2048])]  # only the 4096-wide layer fell back
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = linear._apply(None, layer, x)
    x.mul_(0.5)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), x.float() @ layer.weight.float().t(),
                               atol=2e-2, rtol=1.6e-2)
