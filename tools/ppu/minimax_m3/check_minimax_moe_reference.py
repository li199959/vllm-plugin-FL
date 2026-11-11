"""Independent fp32 reference for M3 MoE boundary batches."""

import json

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
from vllm_fl.optimizations.ppu.minimax_m3.moe import HybridM3Experts


@torch.inference_mode()
def main():
    _disable_flaggems_runtime_autotune()
    for rows in (31, 32, 64):
        torch.manual_seed(43)
        e, h, intermediate, topk = 128, 6144, 192, 4
        activation = MoEActivation.SWIGLUOAI_UNINTERLEAVE
        config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=h,
            intermediate_size=intermediate,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            routing_method=RoutingMethodType.Sigmoid,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
        )
        quant = FusedMoEQuantConfig.make(
            gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0
        )
        x = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16)
        w1 = (
            torch.randn(e, intermediate * 2, h, device="cuda", dtype=torch.bfloat16)
            * 0.05
        )
        w2 = torch.randn(e, h, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02
        ids = torch.topk(torch.randn(rows, e, device="cuda"), topk, dim=-1).indices.to(
            torch.int32
        )
        weights = torch.rand(rows, topk, device="cuda")
        weights = weights / weights.sum(-1, keepdim=True) * 2.0
        expected = torch.zeros(rows, h, device="cuda", dtype=torch.float32)
        for expert in range(e):
            token, slot = torch.where(ids == expert)
            if not token.numel():
                continue
            projected = (
                (x[token].float() @ w1[expert].float().T).to(torch.bfloat16).float()
            )
            gate = projected[:, :intermediate].clamp(max=7.0)
            up = projected[:, intermediate:].clamp(-7.0, 7.0)
            activated = (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(
                torch.bfloat16
            )
            projected2 = (
                (activated.float() @ w2[expert].float().T).to(torch.bfloat16).float()
            )
            expected[token] += projected2 * weights[token, slot, None]
        for cls in (DeepGemmExpertsFL, TritonExpertsFL, HybridM3Experts):
            impl = cls(config, quant)
            impl._lora_context = None
            shapes = impl.workspace_shapes(
                rows, 2 * intermediate, h, topk, e, e, None, activation
            )
            work1, work2, out = [
                torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes
            ]
            for repeat in range(3):
                impl.apply(
                    out,
                    x,
                    w1,
                    w2,
                    weights,
                    ids,
                    activation,
                    e,
                    None,
                    None,
                    None,
                    work1,
                    work2,
                    None,
                    False,
                )
                torch.cuda.synchronize()
                diff = (out.float() - expected).abs()
                bad = diff > (0.08 + expected.abs() * 0.04)
                print(
                    json.dumps(
                        dict(
                            rows=rows,
                            backend=cls.__name__,
                            repeat=repeat,
                            max_abs=diff.max().item(),
                            bad=bad.sum().item(),
                            nan=out.isnan().sum().item(),
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
