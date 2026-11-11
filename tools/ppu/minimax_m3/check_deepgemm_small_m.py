"""Scan native DeepGEMM MoE vs an fp32 reference for every small M (4 inputs x 2 runs)."""

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


@torch.inference_mode()
def main():
    _disable_flaggems_runtime_autotune()
    e, h, inter, topk = 128, 6144, 192, 4
    act = MoEActivation.SWIGLUOAI_UNINTERLEAVE
    cfg = FusedMoEConfig(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=h,
        intermediate_size=inter,
        num_local_experts=e,
        num_logical_experts=e,
        activation=act,
        device="cuda",
        routing_method=RoutingMethodType.Sigmoid,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
    )
    quant = FusedMoEQuantConfig.make(
        gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0
    )
    torch.manual_seed(43)
    w1 = torch.randn(e, inter * 2, h, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(e, h, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    for rows in list(range(1, 65)) + [96, 128]:
        bad_runs = []
        for seed in range(4):
            torch.manual_seed(1000 * rows + seed)
            x = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16)
            ids = torch.topk(
                torch.randn(rows, e, device="cuda"), topk, dim=-1
            ).indices.to(torch.int32)
            wts = torch.rand(rows, topk, device="cuda")
            wts = wts / wts.sum(-1, keepdim=True) * 2.0
            ref = torch.zeros(rows, h, device="cuda")
            for ex in range(e):
                tok, slot = torch.where(ids == ex)
                if not tok.numel():
                    continue
                p = (x[tok].float() @ w1[ex].float().T).to(torch.bfloat16).float()
                g, u = p[:, :inter].clamp(max=7.0), p[:, inter:].clamp(-7.0, 7.0)
                a = (g * torch.sigmoid(1.702 * g) * (u + 1.0)).to(torch.bfloat16)
                ref[tok] += (a.float() @ w2[ex].float().T).to(
                    torch.bfloat16
                ).float() * wts[tok, slot, None]
            impl = DeepGemmExpertsFL(cfg, quant)
            impl._lora_context = None
            shapes = impl.workspace_shapes(rows, 2 * inter, h, topk, e, e, None, act)
            ws1, ws2, out = [
                torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes
            ]
            for rep in range(2):
                out.fill_(float("nan")) if rep else None
                impl.apply(
                    out,
                    x,
                    w1,
                    w2,
                    wts,
                    ids,
                    act,
                    e,
                    None,
                    None,
                    None,
                    ws1,
                    ws2,
                    None,
                    False,
                )
                torch.cuda.synchronize()
                bad = (
                    (out.float() - ref).abs() > 0.08 + 0.04 * ref.abs()
                ) | out.isnan()
                if bad.any():
                    rows_bad = sorted(set(torch.where(bad)[0].tolist()))
                    bad_runs.append(
                        {
                            "seed": seed,
                            "rep": rep,
                            "n": int(bad.sum()),
                            "max": float(
                                (out.float() - ref).abs().nan_to_num(9e9).max()
                            ),
                            "token_rows": rows_bad[:8],
                        }
                    )
        if bad_runs:
            print(json.dumps({"rows": rows, "bad": bad_runs}), flush=True)
    print("scan done", flush=True)


if __name__ == "__main__":
    main()
