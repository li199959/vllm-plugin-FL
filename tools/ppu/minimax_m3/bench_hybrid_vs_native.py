"""Cold-weight timing of the registered hybrid M3 paths vs the original ones.

Times exactly what VLLM_FL_M3_DENSE / VLLM_FL_MOE=hybrid install
(bf16_linear, the fp32 router split-K, HybridM3Experts) against what the
original configuration runs (torch linear, DeepGemmExpertsFL). Weights rotate
through >=1 GiB of copies so every call misses L2, as in a real forward.
"""
import argparse
import json

import torch

from sweep_small_m import measure

# Per-rank linear shapes at TP16: (K, N, dtype).
DENSE = [(6144, 1024, torch.bfloat16), (6144, 384, torch.bfloat16), (512, 6144, torch.bfloat16),
         (192, 6144, torch.bfloat16), (6144, 12504, torch.bfloat16), (6144, 128, torch.float32)]


@torch.inference_mode()
def dense(rows):
    from vllm_fl.optimizations.ppu.minimax_m3.linear import bf16_linear
    from vllm_fl.optimizations.ppu.minimax_m3.splitk import splitk_linear

    for k, n, dtype in DENSE:
        copies = max(8, (1 << 30) // (k * n * dtype.itemsize))
        ws = [torch.randn(n, k, device="cuda", dtype=dtype) * 0.02 for _ in range(copies)]
        hybrid_fn = bf16_linear if dtype == torch.bfloat16 else splitk_linear
        for m in rows:
            x = torch.randn(m, k, device="cuda", dtype=dtype)
            native = measure(lambda i: torch.nn.functional.linear(x, ws[i]), copies)
            hybrid = measure(lambda i: hybrid_fn(x, ws[i], None), copies)
            yield {"kind": "dense", "m": m, "k": k, "n": n, "dtype": str(dtype)[6:],
                   "native_us": native, "hybrid_us": hybrid, "ratio": hybrid / native}
        del ws
        torch.cuda.empty_cache()


@torch.inference_mode()
def moe(rows):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig, RoutingMethodType)

    from vllm_fl.ops.fused_moe.deepgemm_experts import DeepGemmExpertsFL
    from vllm_fl.optimizations.ppu.minimax_m3.moe import HybridM3Experts

    e, h, inter, topk = 128, 6144, 192, 4
    act = MoEActivation.SWIGLUOAI_UNINTERLEAVE
    cfg = FusedMoEConfig(num_experts=e, experts_per_token=topk, hidden_dim=h,
        intermediate_size=inter, num_local_experts=e, num_logical_experts=e,
        activation=act, device="cuda", routing_method=RoutingMethodType.Sigmoid,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(), in_dtype=torch.bfloat16)
    quant = FusedMoEQuantConfig.make(gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0)
    copies = 3
    w1s = [torch.randn(e, inter * 2, h, device="cuda", dtype=torch.bfloat16) * 0.05 for _ in range(copies)]
    w2s = [torch.randn(e, h, inter, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    for m in rows:
        x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
        ids = torch.topk(torch.randn(m, e, device="cuda"), topk, dim=-1).indices.to(torch.int32)
        wts = torch.rand(m, topk, device="cuda")

        def prepare(cls):
            impl = cls(cfg, quant)
            impl._lora_context = None
            shapes = impl.workspace_shapes(m, 2 * inter, h, topk, e, e, None, act)
            ws1, ws2, out = [torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes]
            return lambda i: impl.apply(out, x, w1s[i], w2s[i], wts, ids, act, e, None, None,
                                        None, ws1, ws2, None, False)

        native = measure(prepare(DeepGemmExpertsFL), copies)
        hybrid = measure(prepare(HybridM3Experts), copies)
        yield {"kind": "moe", "m": m, "native_us": native, "hybrid_us": hybrid,
               "ratio": hybrid / native}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 32, 48, 64, 96, 128])
    parser.add_argument("--section", choices=("dense", "moe", "all"), default="all")
    args = parser.parse_args()
    torch.manual_seed(7)
    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune

    _disable_flaggems_runtime_autotune()
    sections = {"dense": (dense,), "moe": (moe,), "all": (dense, moe)}[args.section]
    for gen in sections:
        for row in gen(args.rows):
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
