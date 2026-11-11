#!/usr/bin/env python3
"""Qwen3-VL MoE (TP2 shard) per layer: tuned Gems fused_experts vs DeepGEMM nopad.

DeepGEMM path mirrors vllm_fl DeepGemmExpertsFL.apply (permute, two grouped
GEMMs, SiLU-and-mul, weighted gather). Both are checked against an FP32
reference.

  CUDA_VISIBLE_DEVICES=3 python3 bench_moe_deepgemm.py --rows 256 1024 4096 8192
"""

import argparse
import sys

import torch

E, HIDDEN, INTER, TOPK = 128, 2048, 384, 8


def _time(fn, iters=20):
    for _ in range(3):
        fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


def _reference(x, w1, w2, tw, ids):
    out = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    flat = ids.reshape(-1).long()
    tokens = torch.arange(x.shape[0], device="cuda").repeat_interleave(TOPK)
    weights = tw.reshape(-1)
    for e in flat.unique().tolist():
        sel = (flat == e).nonzero().squeeze(1)
        t = tokens[sel]
        gu = x[t].float() @ w1[e].float().t()
        act = (
            (torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]).bfloat16().float()
        )
        out.index_add_(0, t, weights[sel, None] * (act @ w2[e].float().t()))
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rows", type=int, nargs="+", default=[64, 256, 1024, 2048, 4096, 8192]
    )
    a = ap.parse_args()

    import deep_gemm
    import flag_gems  # noqa: F401
    from deep_gemm.deep_gemm_tuner.deepgemm_tools import deepgemm_moe_permute

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_gather

    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune
    from vllm_fl.ops.fused_moe.activation import apply_moe_activation
    from vllm_fl.optimizations.ppu.minimax_m3.moe import disable_deepgemm_gemv
    from vllm_fl.optimizations.ppu.qwen3_vl_moe import moe

    _disable_flaggems_runtime_autotune()
    disable_deepgemm_gemv()  # small-K GEMV is wrong on PPU (see minimax_m3 README)
    moe.install_tuned_moe_tiles()
    fm = sys.modules["flag_gems.fused.fused_moe"]
    torch.manual_seed(0)
    w1 = (torch.randn(E, 2 * INTER, HIDDEN, device="cuda") * 0.02).bfloat16()
    w2 = (torch.randn(E, HIDDEN, INTER, device="cuda") * 0.02).bfloat16()
    for m in a.rows:
        x = (torch.randn(m, HIDDEN, device="cuda") * 0.5).bfloat16()
        w, ids = torch.topk(torch.randn(m, E, device="cuda").softmax(-1), TOPK, dim=-1)
        tw, ids = (w / w.sum(-1, keepdim=True)).float(), ids.to(torch.int32)
        out_dg = torch.empty_like(x)
        mm1 = torch.empty(m * TOPK, 2 * INTER, device="cuda", dtype=torch.bfloat16)
        act = torch.empty(m * TOPK, INTER, device="cuda", dtype=torch.bfloat16)
        mm2 = torch.empty(m * TOPK, HIDDEN, device="cuda", dtype=torch.bfloat16)

        def gems():
            return fm.fused_experts_impl(x, w1, w2, tw, ids)

        def dg():
            a1, _, m_idx, inv, m_rows = deepgemm_moe_permute(
                x, None, ids, E, block_align=1, block_k=HIDDEN
            )
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(a1, w1, mm1, m_idx, m_rows)
            apply_moe_activation(MoEActivation.SILU, act, mm1)
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
                act, w2, mm2, m_idx, m_rows
            )
            ep_gather(
                input_tensor=mm2,
                recv_topk_ids=ids,
                recv_topk_weight=tw,
                input_index=inv,
                expert_map=None,
                output_tensor=out_dg,
            )
            return out_dg

        ref = _reference(x, w1, w2, tw, ids)
        err_g = (gems().float() - ref).abs().max().item()
        err_d = (dg().float() - ref).abs().max().item()
        g_us, d_us = _time(gems), _time(dg)
        print(
            f"M={m:5d} gems {g_us:8.1f} us  deepgemm {d_us:8.1f} us  (gems/dg {g_us / d_us:.2f})  "
            f"err gems {err_g:.2e} dg {err_d:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
