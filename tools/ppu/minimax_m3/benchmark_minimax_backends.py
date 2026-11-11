"""Compare native and Gems M3 matmuls on an idle GPU using CUDA graphs."""

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
import triton


def measure(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fn()
    for _ in range(3):
        graph.replay()
    values = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 10)
    return statistics.median(values)  # microseconds per call


@torch.inference_mode()
def dense(rows):
    import flag_gems
    from flag_gems.ops.linear import linear_kernel

    raw = linear_kernel
    while not isinstance(raw, triton.runtime.JITFunction):
        raw = raw.fn
    print("Gems linear entry:", flag_gems.linear.__module__, flush=True)
    for m in rows:
        for k, n, dtype in [
            (6144, 1024, torch.bfloat16),
            (6144, 384, torch.bfloat16),
            (512, 6144, torch.bfloat16),
            (192, 6144, torch.bfloat16),
            (6144, 128, torch.float32),
            (6144, 12504, torch.bfloat16),
        ]:
            if m > 64 and n == 12504:
                continue
            x = torch.randn(m, k, device="cuda", dtype=dtype)
            w = torch.randn(n, k, device="cuda", dtype=dtype) * 0.02
            out = torch.empty((m, n), device="cuda", dtype=dtype)
            native = lambda: torch.nn.functional.linear(x, w)
            expected = native()
            for name, fn in [
                ("native", native),
                ("gems_default", lambda: flag_gems.linear(x, w)),
            ]:
                actual = fn()
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)
                yield {
                    "kind": "dense",
                    "shape": [m, n, k],
                    "dtype": str(dtype),
                    "backend": name,
                    "us": measure(fn),
                }
            if dtype == torch.float32:
                continue  # Retain native routing precision and performance.
            configs = (
                [(32, 64, 128, 4, 3), (16, 64, 128, 4, 3), (64, 64, 64, 4, 3)]
                if m <= 64
                else [(64, 128, 64, 4, 3), (128, 128, 64, 8, 3)]
            )
            for bm, bn, bk, warps, stages in configs:

                def run():
                    raw[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                        x,
                        w,
                        w,
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
                    return out

                torch.testing.assert_close(run(), expected, rtol=0.03, atol=0.05)
                yield {
                    "kind": "dense",
                    "shape": [m, n, k],
                    "dtype": str(dtype),
                    "backend": "gems_direct",
                    "config": [bm, bn, bk, warps, stages],
                    "us": measure(run),
                }


@torch.inference_mode()
def moe(rows):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )

    import vllm_fl.ops.fused_moe.fused_moe_utils as utils
    from vllm_fl.ops.fused_moe.deepgemm_experts import DeepGemmExpertsFL

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
    w1 = torch.randn(e, intermediate * 2, h, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(e, h, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02
    original_config = utils.try_get_optimal_moe_config
    for m in rows:
        x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
        ids = torch.topk(torch.randn(m, e, device="cuda"), topk, dim=-1).indices.to(
            torch.int32
        )
        weights = torch.rand(m, topk, device="cuda")
        weights = weights / weights.sum(-1, keepdim=True) * 2.0

        def prepare(cls):
            impl = cls(config, quant)
            impl._lora_context = None
            shapes = impl.workspace_shapes(
                m, 2 * intermediate, h, topk, e, e, None, activation
            )
            work1, work2, out = [
                torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes
            ]

            def run():
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
                return out

            return run

        native, gems = prepare(DeepGemmExpertsFL), prepare(utils.TritonExpertsFL)
        expected = native().clone()
        default_config = original_config(w1.shape, w2.shape, topk, None, m)
        for name, fn in [("native", native), ("gems_default", gems)]:
            torch.testing.assert_close(fn(), expected, rtol=0.04, atol=0.08)
            yield {
                "kind": "moe",
                "rows": m,
                "backend": name,
                "config": default_config,
                "us": measure(fn),
            }
        tiles = (
            [
                (16, 64, 128, 4, 3),
                (16, 128, 64, 4, 3),
                (16, 64, 64, 4, 3),
                (16, 128, 128, 4, 3),
            ]
            if m <= 64
            else [(64, 128, 64, 4, 3), (64, 64, 64, 4, 3), (32, 128, 64, 4, 3)]
        )
        try:
            for bm, bn, bk, warps, stages in tiles:
                tuned = dict(
                    BLOCK_SIZE_M=bm,
                    BLOCK_SIZE_N=bn,
                    BLOCK_SIZE_K=bk,
                    GROUP_SIZE_M=1,
                    SPLIT_K=1,
                    num_warps=warps,
                    num_stages=stages,
                )
                utils.try_get_optimal_moe_config = lambda *a, **kw: tuned
                torch.testing.assert_close(gems(), expected, rtol=0.04, atol=0.08)
                yield {
                    "kind": "moe",
                    "rows": m,
                    "backend": "gems_tuned",
                    "config": tuned,
                    "us": measure(gems),
                }
        finally:
            utils.try_get_optimal_moe_config = original_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("dense", "moe"), required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 8192])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(43)
    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune

    _disable_flaggems_runtime_autotune()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for result in (dense if args.section == "dense" else moe)(args.rows):
            line = json.dumps(result)
            print(line, flush=True)
            stream.write(line + "\n")
            stream.flush()
    gc.collect()


if __name__ == "__main__":
    main()
