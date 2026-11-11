"""HybridQwen3VLExperts: tuned Gems below DEEPGEMM_MIN_TOKENS, DeepGEMM at/above."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)

from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune
from vllm_fl.ops.fused_moe.deepgemm_experts import DeepGemmExpertsFL
from vllm_fl.optimizations.ppu.qwen3_vl_moe import moe

pytestmark = pytest.mark.gpu

E, HIDDEN, INTER, TOPK = 128, 2048, 384, 8


def _reference(x, w1, w2, weights, ids):
    out = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    for e in ids.unique().tolist():
        token, slot = torch.where(ids == e)
        gu = x[token].float() @ w1[e].float().t()
        act = (
            (torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]).bfloat16().float()
        )
        out[token] += weights[token, slot, None] * (act @ w2[e].float().t())
    return out


@pytest.mark.parametrize("rows", [64, 1024, 4100, 8191, 8192])
@torch.inference_mode()
def test_hybrid_experts_route_and_match_reference(monkeypatch, rows):
    _disable_flaggems_runtime_autotune()
    moe.install_tuned_moe_tiles()
    cls = moe.hybrid_experts_class()
    calls = []
    original = DeepGemmExpertsFL.apply
    monkeypatch.setattr(
        DeepGemmExpertsFL,
        "apply",
        lambda self, *a, **kw: calls.append(1) or original(self, *a, **kw),
    )
    config = FusedMoEConfig(
        num_experts=E,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        intermediate_size=INTER,
        num_local_experts=E,
        num_logical_experts=E,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
    )
    impl = cls(config, FusedMoEQuantConfig.make())
    impl._lora_context = None
    torch.manual_seed(rows)
    x = (torch.randn(rows, HIDDEN, device="cuda") * 0.5).bfloat16()
    w1 = (torch.randn(E, 2 * INTER, HIDDEN, device="cuda") * 0.02).bfloat16()
    w2 = (torch.randn(E, HIDDEN, INTER, device="cuda") * 0.02).bfloat16()
    w, ids = torch.topk(torch.randn(rows, E, device="cuda").softmax(-1), TOPK, dim=-1)
    weights, ids = (w / w.sum(-1, keepdim=True)).float(), ids.to(torch.int32)
    shapes = impl.workspace_shapes(
        rows, 2 * INTER, HIDDEN, TOPK, E, E, None, MoEActivation.SILU
    )
    work1, work2, out = [
        torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes
    ]
    impl.apply(
        out,
        x,
        w1,
        w2,
        weights,
        ids,
        MoEActivation.SILU,
        E,
        None,
        None,
        None,
        work1,
        work2,
        None,
        False,
    )
    assert len(calls) == (1 if rows >= moe.DEEPGEMM_MIN_TOKENS else 0)
    torch.testing.assert_close(
        out.float(), _reference(x, w1, w2, weights, ids), atol=3e-3, rtol=3e-2
    )
