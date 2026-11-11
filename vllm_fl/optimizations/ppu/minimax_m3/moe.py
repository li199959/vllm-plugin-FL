# Copyright (c) 2026 BAAI. All rights reserved.
"""Hybrid DeepGEMM / Gems experts for MiniMax-M3 (VLLM_FL_MOE=hybrid).

Tuned for BF16, E=128, topk=4, K=6144, I=192 (per-rank shapes at TP16).
"""

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.ops.fused_moe.deepgemm_experts import DeepGemmExpertsFL
from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL

_gemv_disabled = False


def disable_deepgemm_gemv() -> None:
    """Disable the PPU small-K GEMV path, which can return incorrect values.

    ThreadPerN == -1 is DeepGEMM's configuration for grouped GEMM without GEMV.
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
        # Decode uses DeepGEMM grouped GEMM; prefill uses FlagGems.
        if hidden_states.shape[0] <= 128:
            return DeepGemmExpertsFL.apply(self, output, hidden_states, *args, **kwargs)
        return super().apply(output, hidden_states, *args, **kwargs)

    def tuned_config(self, E, num_tokens, N, K, top_k_num, dtype, activation):
        import torch

        assert (E, K, N, top_k_num) == (128, 6144, 384, 4)
        assert dtype == torch.bfloat16
        assert activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
        # Fixed tile configurations for the supported M3 shard.
        if num_tokens <= 512:
            bm, bn, bk, stages = 32, 128, 64, 3
        else:
            bm, bn, bk, stages = 64, 128, 64, 3
        return dict(
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            GROUP_SIZE_M=1,
            SPLIT_K=1,
            num_warps=4,
            num_stages=stages,
        )
