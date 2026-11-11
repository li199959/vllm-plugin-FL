"""Sweep Gems tiles for M3 decode shapes with cold (L2-missing) weights.

Each timed call walks a ring of weight copies totalling >= 1 GiB, as a real
forward does across layers, so the result reflects HBM-bound behaviour.
"""

import argparse
import itertools
import json
import statistics

import torch
import triton

DENSE = [(6144, 1024), (6144, 384), (512, 6144), (192, 6144), (6144, 12504)]


def measure(step, copies):
    for i in range(copies):
        step(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(copies):
            step(i)
    graph.replay()
    values = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / copies)
    return statistics.median(values)


@torch.inference_mode()
def dense(rows):
    from flag_gems.ops.linear import linear_kernel

    raw = linear_kernel
    while not isinstance(raw, triton.runtime.JITFunction):
        raw = raw.fn
    for k, n in DENSE:
        copies = max(8, (1 << 30) // (k * n * 2))
        ws = [
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
            for _ in range(copies)
        ]
        for m in rows:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            native = measure(lambda i: torch.nn.functional.linear(x, ws[i]), copies)
            best = None
            for bm, bn, bk, warps, stages in itertools.product(
                (16, 32, 64), (32, 64, 128), (64, 128, 256), (4, 8), (2, 3, 4)
            ):
                if bm > max(16, triton.next_power_of_2(m)) * 2:
                    continue

                def step(i):
                    raw[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                        x,
                        ws[i],
                        ws[i],
                        out,
                        m,
                        n,
                        k,
                        k,
                        1,
                        k,
                        1,
                        n,
                        1,
                        0,
                        False,
                        bm,
                        bn,
                        bk,
                        num_warps=warps,
                        num_stages=stages,
                    )

                try:
                    us = measure(step, copies)
                except Exception:  # resource limits on some tiles
                    continue
                if best is None or us < best[0]:
                    best = (us, [bm, bn, bk, warps, stages])
            torch.testing.assert_close(
                out, torch.nn.functional.linear(x, ws[-1]), rtol=0.03, atol=0.05
            )
            yield {
                "kind": "dense",
                "m": m,
                "k": k,
                "n": n,
                "native_us": native,
                "best_us": best[0],
                "best": best[1],
                "ratio": best[0] / native,
            }
        del ws
        torch.cuda.empty_cache()


@torch.inference_mode()
def moe(rows):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )

    from vllm_fl.ops.fused_moe.deepgemm_experts import DeepGemmExpertsFL
    from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL

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
    copies = 3  # ~600 MiB of expert weights each, already far beyond L2
    w1s = [
        torch.randn(e, inter * 2, h, device="cuda", dtype=torch.bfloat16) * 0.05
        for _ in range(copies)
    ]
    w2s = [
        torch.randn(e, h, inter, device="cuda", dtype=torch.bfloat16) * 0.02
        for _ in range(copies)
    ]

    class Swept(TritonExpertsFL):
        tile = None

        def tuned_config(self, *args):
            bm, bn, bk, warps, stages = self.tile
            return dict(
                BLOCK_SIZE_M=bm,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=bk,
                GROUP_SIZE_M=1,
                SPLIT_K=1,
                num_warps=warps,
                num_stages=stages,
            )

    for m in rows:
        x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
        ids = torch.topk(torch.randn(m, e, device="cuda"), topk, dim=-1).indices.to(
            torch.int32
        )
        wts = torch.rand(m, topk, device="cuda")

        def prepare(impl):
            impl._lora_context = None
            shapes = impl.workspace_shapes(m, 2 * inter, h, topk, e, e, None, act)
            ws1, ws2, out = [
                torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes
            ]
            step = lambda i: impl.apply(
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
            )
            return step, out

        native_step, native_out = prepare(DeepGemmExpertsFL(cfg, quant))
        native = measure(native_step, copies)
        best = None
        for tile in itertools.product(
            (16, 32), (32, 64, 128), (64, 128, 256), (4, 8), (2, 3, 4)
        ):
            impl = Swept(cfg, quant)
            impl.tile = tile
            step, out = prepare(impl)
            try:
                us = measure(step, copies)
            except Exception:
                continue
            if best is None or us < best[0]:
                best = (us, list(tile))
        yield {
            "kind": "moe",
            "m": m,
            "native_us": native,
            "best_us": best[0],
            "best": best[1],
            "ratio": best[0] / native,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("dense", "moe"), required=True)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 8, 16, 32, 48, 64, 96, 128]
    )
    args = parser.parse_args()
    torch.manual_seed(11)
    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune

    _disable_flaggems_runtime_autotune()
    for row in (dense if args.section == "dense" else moe)(args.rows):
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
