"""Exercise the existing FlagGems expert path with M3's per-rank shapes."""

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
from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL
from vllm_fl.optimizations.ppu.minimax_m3.moe import HybridM3Experts, disable_deepgemm_gemv

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("rows", [1, 31, 32, 48, 49, 63, 64, 128, 129, 257])
@pytest.mark.parametrize("expert_cls", [TritonExpertsFL, HybridM3Experts])
@torch.inference_mode()
def test_m3_gems_experts_native_and_graph(rows, expert_cls):
    _disable_flaggems_runtime_autotune()
    torch.manual_seed(43)
    experts, hidden, intermediate, topk = 128, 6144, 192, 4
    activation = MoEActivation.SWIGLUOAI_UNINTERLEAVE
    config = FusedMoEConfig(
        num_experts=experts, experts_per_token=topk, hidden_dim=hidden,
        intermediate_size=intermediate, num_local_experts=experts,
        num_logical_experts=experts, activation=activation, device="cuda",
        routing_method=RoutingMethodType.Sigmoid,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
    )
    quant = FusedMoEQuantConfig.make(
        gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0,
    )
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts, intermediate * 2, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02
    ids = torch.topk(torch.randn(rows, experts, device="cuda"), topk, dim=-1).indices.to(torch.int32)
    weights = torch.rand(rows, topk, device="cuda")
    weights = weights / weights.sum(-1, keepdim=True) * 2.0

    def prepare(cls):
        impl = cls(config, quant)
        impl._lora_context = None
        shapes = impl.workspace_shapes(rows, 2 * intermediate, hidden, topk, experts, experts, None, activation)
        work1, work2, out = [torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes]

        def run():
            impl.apply(out, x, w1, w2, weights, ids, activation, experts,
                       None, None, None, work1, work2, None, False)
            return out

        return run

    if expert_cls is HybridM3Experts:
        disable_deepgemm_gemv()
    native, gems = prepare(DeepGemmExpertsFL), prepare(expert_cls)
    # Independent fp32 reference for every token. Stock DeepGEMM small-K GEMV
    # (rows <= 47) is wrong; with it disabled native must match at every size.
    expected = torch.zeros(rows, hidden, device="cuda", dtype=torch.float32)
    for expert in range(experts):
        token, slot = torch.where(ids == expert)
        if not token.numel():
            continue
        projected = (x[token].float() @ w1[expert].float().T).to(torch.bfloat16).float()
        gate = projected[:, :intermediate].clamp(max=7.0)
        up = projected[:, intermediate:].clamp(-7.0, 7.0)
        activated = (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(torch.bfloat16)
        projected2 = (activated.float() @ w2[expert].float().T).to(torch.bfloat16).float()
        expected[token] += projected2 * weights[token, slot, None]
    if rows >= 49 or expert_cls is HybridM3Experts:
        torch.testing.assert_close(native().float(), expected, rtol=0.04, atol=0.08)
    actual = gems()
    torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.08)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gems()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gems()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.08)
