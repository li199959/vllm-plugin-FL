"""Pick small-M Gems MoE tiles robustly: mean over several routing seeds."""

import argparse
import json
import statistics

import torch
from sweep_small_m import measure

SMALL = [
    (16, 64, 64, 3, 4),
    (16, 128, 64, 2, 4),
    (16, 128, 64, 3, 4),
    (16, 64, 128, 4, 4),
    (16, 128, 128, 2, 4),
    (32, 128, 64, 2, 4),
]
LARGE = [
    (bm, bn, bk, st, w)
    for bm in (32, 64, 128)
    for bn in (64, 128, 256)
    for bk in (32, 64)
    for st in (2, 3)
    for w in (4, 8)
    if bm * bn <= 128 * 128 and not (bm == 128 and bn == 256)
]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 4, 8, 16, 24, 32, 40, 48]
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--large", action="store_true", help="prefill-size candidate grid"
    )
    args = parser.parse_args()
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
    copies = 3
    w1s = [
        torch.randn(e, inter * 2, h, device="cuda", dtype=torch.bfloat16) * 0.05
        for _ in range(copies)
    ]
    w2s = [
        torch.randn(e, h, inter, device="cuda", dtype=torch.bfloat16) * 0.02
        for _ in range(copies)
    ]

    class Fixed(TritonExpertsFL):
        tile = None

        def tuned_config(self, *args):
            bm, bn, bk, stages, warps = self.tile
            return dict(
                BLOCK_SIZE_M=bm,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=bk,
                GROUP_SIZE_M=1,
                SPLIT_K=1,
                num_warps=warps,
                num_stages=stages,
            )

    def timed(impl, m, x, ids, wts):
        impl._lora_context = None
        shapes = impl.workspace_shapes(m, 2 * inter, h, topk, e, e, None, act)
        ws1, ws2, out = [
            torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes
        ]
        return measure(
            lambda i: impl.apply(
                out,
                x,
                w1s[i],
                w2s[i],
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
            ),
            copies,
        )

    candidates = LARGE if args.large else SMALL
    for m in args.rows:
        native, per_tile = [], {t: [] for t in candidates}
        for seed in range(args.seeds):
            torch.manual_seed(100 + seed)
            x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
            ids = torch.topk(torch.randn(m, e, device="cuda"), topk, dim=-1).indices.to(
                torch.int32
            )
            wts = torch.rand(m, topk, device="cuda")
            native.append(timed(DeepGemmExpertsFL(cfg, quant), m, x, ids, wts))
            for tile in candidates:
                impl = Fixed(cfg, quant)
                impl.tile = tile
                try:
                    per_tile[tile].append(timed(impl, m, x, ids, wts))
                except Exception:  # resource limits on some tiles
                    per_tile[tile].append(float("inf"))
        means = {t: statistics.mean(v) for t, v in per_tile.items()}
        best = min(means, key=means.get)
        print(
            json.dumps(
                {
                    "m": m,
                    "native_us": statistics.mean(native),
                    "tiles": {str(list(t)): round(v, 1) for t, v in means.items()},
                    "best": list(best),
                    "ratio": means[best] / statistics.mean(native),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
