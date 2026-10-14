"""Device memory each M3 path costs outside the torch allocator.

Runs each path once at prefill size (M=8192 per rank) and reports how much
device memory disappeared beyond what torch reserved: loaded CUDA modules,
local-memory reservations for spilling kernels, library workspaces. vLLM sizes
the KV cache after such allocations, so they reduce KV capacity directly.
"""
import argparse
import json

import torch


def nontorch_delta(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free0, _ = torch.cuda.mem_get_info()
    res0 = torch.cuda.memory_reserved()
    fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free1, _ = torch.cuda.mem_get_info()
    res1 = torch.cuda.memory_reserved()
    return ((free0 - free1) - (res1 - res0)) / 2**20


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8192)
    args = parser.parse_args()
    m, h = args.m, 6144
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig, RoutingMethodType)

    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune

    _disable_flaggems_runtime_autotune()
    torch.zeros(1, device="cuda")  # context
    results = {}
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    e, inter, topk = 128, 192, 4
    act = MoEActivation.SWIGLUOAI_UNINTERLEAVE
    cfg = FusedMoEConfig(num_experts=e, experts_per_token=topk, hidden_dim=h,
        intermediate_size=inter, num_local_experts=e, num_logical_experts=e,
        activation=act, device="cuda", routing_method=RoutingMethodType.Sigmoid,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(), in_dtype=torch.bfloat16)
    quant = FusedMoEQuantConfig.make(gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0)
    w1 = torch.randn(e, inter * 2, h, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(e, h, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    ids = torch.topk(torch.randn(m, e, device="cuda"), topk, dim=-1).indices.to(torch.int32)
    wts = torch.rand(m, topk, device="cuda")

    def experts(cls, rows):
        def run():
            impl = cls(cfg, quant)
            impl._lora_context = None
            shapes = impl.workspace_shapes(rows, 2 * inter, h, topk, e, e, None, act)
            ws1, ws2, out = [torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes]
            impl.apply(out, x[:rows], w1, w2, wts[:rows], ids[:rows], act, e, None, None, None,
                       ws1, ws2, None, False)
        return run

    from vllm_fl.ops.fused_moe.deepgemm_experts import DeepGemmExpertsFL
    from vllm_fl.optimizations.ppu.minimax_m3.moe import HybridM3Experts

    # Order matters only for shared libraries; each path is measured on first use.
    results["native_deepgemm_moe_prefill"] = nontorch_delta(experts(DeepGemmExpertsFL, m))
    results["native_deepgemm_moe_decode"] = nontorch_delta(experts(DeepGemmExpertsFL, 64))
    results["gems_moe_prefill"] = nontorch_delta(experts(HybridM3Experts, m))
    results["gems_moe_small"] = nontorch_delta(experts(HybridM3Experts, 16))

    from vllm_fl.optimizations.ppu.minimax_m3.norm import gemma_fused_add_rms_norm, gemma_rms_norm
    wn = torch.randn(h, device="cuda", dtype=torch.bfloat16)
    res = torch.randn_like(x)
    results["gems_gemma_norms"] = nontorch_delta(
        lambda: (gemma_rms_norm(x, wn, 1e-6), gemma_fused_add_rms_norm(x.clone(), res.clone(), wn, 1e-6)))

    from vllm_fl.optimizations.ppu.minimax_m3.linear import bf16_linear
    from vllm_fl.optimizations.ppu.minimax_m3.splitk import splitk_linear
    wl = torch.randn(1024, h, device="cuda", dtype=torch.bfloat16)
    results["hybrid_linear_decode"] = nontorch_delta(
        lambda: [bf16_linear(x[:r], wl, None) for r in (1, 16, 64, 128)])
    wr = torch.randn(128, h, device="cuda")
    results["splitk_router"] = nontorch_delta(lambda: splitk_linear(x[:64].float(), wr, None))

    import flag_gems
    results["gems_index"] = nontorch_delta(
        lambda: flag_gems.index(x, [torch.arange(0, m, 2, device="cuda")]))
    print(json.dumps({k: round(v, 1) for k, v in results.items()}, indent=1))
    print(json.dumps({"total_reserved_MiB": torch.cuda.memory_reserved() / 2**20}))


if __name__ == "__main__":
    main()
