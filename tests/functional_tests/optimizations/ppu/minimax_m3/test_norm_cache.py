import pytest
import torch

from vllm_fl.optimizations.ppu.minimax_m3 import norm
from vllm_fl.optimizations.ppu.minimax_m3.norm import gemma_fused_add_rms_norm

pytestmark = pytest.mark.gpu


def _reference(x, residual, weight, eps):
    h = x.float() + residual.float()
    out = (
        h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())
    )
    return out.to(x.dtype), h.to(x.dtype)


@torch.no_grad()
def test_shifted_weight_cached_and_refreshed():
    torch.manual_seed(0)
    # Regular (non-inference) tensor: in-place updates bump its version.
    w = torch.randn(6144, device="cuda", dtype=torch.bfloat16) * 0.1
    first = norm._shifted_weight(w)
    assert norm._shifted_weight(w) is first
    torch.testing.assert_close(first, w.float() + 1.0, rtol=0, atol=0)
    w.add_(1.0)  # in-place weight update (e.g. reload) must not reuse stale cache
    assert norm._shifted_weight(w) is not first
    torch.testing.assert_close(norm._shifted_weight(w), w.float() + 1.0, rtol=0, atol=0)


@torch.inference_mode()
def test_shifted_weight_inference_tensor():
    w = torch.randn(6144, device="cuda", dtype=torch.bfloat16) * 0.1
    first = norm._shifted_weight(w)
    assert norm._shifted_weight(w) is first
    torch.testing.assert_close(first, w.float() + 1.0, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 64, 300])
@torch.inference_mode()
def test_fused_add_norm_graph_with_cache(rows):
    torch.manual_seed(rows)
    w = torch.randn(6144, device="cuda", dtype=torch.bfloat16) * 0.1
    x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    want_out, want_res = _reference(x, r, w, 1e-6)
    xs, rs = x.clone(), r.clone()
    gemma_fused_add_rms_norm(xs, rs, w, 1e-6)  # eager first use fills the cache
    torch.testing.assert_close(xs, want_out, rtol=0.02, atol=0.02)
    torch.testing.assert_close(rs, want_res, rtol=0.02, atol=0.02)
    xg, rg = x.clone(), r.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemma_fused_add_rms_norm(xg, rg, w, 1e-6)
    xg.copy_(x)
    rg.copy_(r)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(xg, xs, rtol=0, atol=0)
    torch.testing.assert_close(rg, rs, rtol=0, atol=0)
