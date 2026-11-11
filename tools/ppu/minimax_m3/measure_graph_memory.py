"""CUDA-graph pool memory per M3 op: capture 60 calls (one per layer) and
report the private pool size. An op whose temporaries are not reused inside
the capture grows ~60x its per-call allocation; vLLM reserves that memory
before sizing the KV cache.
"""

import argparse
import json

import torch

LAYERS = 60


def pool_mib(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    fn()  # warm up / compile outside capture
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(LAYERS):
            fn()
    torch.cuda.synchronize()
    used = (torch.cuda.memory_reserved() - before) / 2**20
    del graph
    torch.cuda.empty_cache()
    return round(used, 1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    args = parser.parse_args()
    m, h = args.m, 6144
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )

    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune
    from vllm_fl.ops.fused_moe.deepgemm_experts import DeepGemmExpertsFL
    from vllm_fl.optimizations.ppu.common.shape_stable import install_shape_stable_gems
    from vllm_fl.optimizations.ppu.minimax_m3.linear import bf16_linear
    from vllm_fl.optimizations.ppu.minimax_m3.moe import HybridM3Experts
    from vllm_fl.optimizations.ppu.minimax_m3.norm import (
        gemma_fused_add_rms_norm,
        gemma_rms_norm,
    )
    from vllm_fl.optimizations.ppu.minimax_m3.splitk import splitk_linear

    _disable_flaggems_runtime_autotune()
    install_shape_stable_gems()
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    res = torch.randn_like(x)
    wn = torch.randn(h, device="cuda", dtype=torch.bfloat16)
    wl = torch.randn(1024, h, device="cuda", dtype=torch.bfloat16)
    wr = torch.randn(128, h, device="cuda")
    xr = x.float()
    out = {"m": m, "per_call_output_MiB": round(m * h * 2 / 2**20, 2)}
    out["torch_linear_bf16"] = pool_mib(
        lambda: torch.nn.functional.linear(x[:, :h], wl)
    )
    out["hybrid_bf16_linear"] = pool_mib(lambda: bf16_linear(x, wl, None))
    out["torch_linear_fp32_router"] = pool_mib(
        lambda: torch.nn.functional.linear(xr, wr)
    )
    out["splitk_router"] = pool_mib(lambda: splitk_linear(xr, wr, None))
    out["gems_gemma_rms_norm"] = pool_mib(lambda: gemma_rms_norm(x, wn, 1e-6))
    out["gems_fused_add_rms_norm"] = pool_mib(
        lambda: gemma_fused_add_rms_norm(x, res, wn, 1e-6)
    )
    import flag_gems

    idx = torch.arange(0, m, 2, device="cuda")
    out["gems_index"] = pool_mib(lambda: flag_gems.index(x, [idx]))

    e, inter, topk = 128, 192, 4
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
    w1 = torch.randn(e, inter * 2, h, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(e, h, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    ids = torch.topk(torch.randn(m, e, device="cuda"), topk, dim=-1).indices.to(
        torch.int32
    )
    wts = torch.rand(m, topk, device="cuda")
    for name, cls in (
        ("deepgemm_moe", DeepGemmExpertsFL),
        ("hybrid_moe", HybridM3Experts),
    ):
        impl = cls(cfg, quant)
        impl._lora_context = None
        shapes = impl.workspace_shapes(m, 2 * inter, h, topk, e, e, None, act)
        ws1, ws2, o = [
            torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes
        ]
        out[name] = pool_mib(
            lambda: impl.apply(
                o, x, w1, w2, wts, ids, act, e, None, None, None, ws1, ws2, None, False
            )
        )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
