# Copyright (c) 2026 BAAI. All rights reserved.
"""Hybrid DeepGEMM / Gems experts for MiniMax-M3 (VLLM_FL_MOE=hybrid).

Tuned for BF16, E=128, topk=4, K=6144, I=192 (per-rank shapes at TP16).
"""

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.ops.fused_moe.deepgemm_experts import DeepGemmExpertsFL
from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL

_gemv_disabled = False


def disable_deepgemm_gemv() -> None:
    """Make DeepGEMM's nopad grouped GEMM never pick its GEMV kernels.

    For small M it auto-selects GEMV; the small-K variant (gemm2, K=192,
    M<=47 here) returns wrong, run-to-run varying elements (abs error ~1 on
    outputs of ~0.3, tools/ppu/minimax_m3/check_deepgemm_gemv.py). The
    grouped GEMM it falls back to is correct at every M and within a few
    percent of GEMV speed. ThreadPerN == -1 is DeepGEMM's own "no GEMV" value.
    """
    global _gemv_disabled
    if _gemv_disabled:
        return
    from deep_gemm.jit_kernels import m_grouped_gemm as grouped

    original = grouped.get_gemv_best_configs

    def no_gemv(*args, **kwargs):
        config = list(original(*args, **kwargs))
        config[1] = -1
        return tuple(config)

    grouped.get_gemv_best_configs = no_gemv
    _gemv_disabled = True


class HybridM3Experts(TritonExpertsFL):
    # Both paths reuse these buffers; the compact native layout is at least
    # as large as the modular Gems scratch space (M * topk * max(K, 2I)).
    workspace_shapes = DeepGemmExpertsFL.workspace_shapes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        disable_deepgemm_gemv()

    def apply(self, output, hidden_states, *args, **kwargs):
        # Decode (<=128 tokens): DeepGEMM grouped GEMM, 1.2-1.3x faster than
        # the best Gems tiles with cold weights. Prefill: Gems.
        if hidden_states.shape[0] <= 128:
            return DeepGemmExpertsFL.apply(self, output, hidden_states, *args, **kwargs)
        return super().apply(output, hidden_states, *args, **kwargs)

    def tuned_config(self, E, num_tokens, N, K, top_k_num, dtype, activation):
        import torch

        assert (E, K, N, top_k_num) == (128, 6144, 384, 4)
        assert dtype == torch.bfloat16
        assert activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
        # Best of a 64-tile cold-weight sweep (sweep_moe_tiles.py --large):
        # 1.07-1.4x DeepGEMM, kept on Gems for coverage.
        if num_tokens <= 512:
            bm, bn, bk, stages = 32, 128, 64, 3
        else:
            bm, bn, bk, stages = 64, 128, 64, 3
        return dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
                    SPLIT_K=1, num_warps=4, num_stages=stages)
