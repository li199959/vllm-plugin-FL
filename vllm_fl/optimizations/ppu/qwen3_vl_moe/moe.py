"""Tuned Gems fused-MoE tiles for Qwen3-VL-30B-A3B on PPU (TP2 shard).

Gems has no embedded tile table for PPU and falls back to get_default_config.
For E=128, hidden 2048, per-rank intermediate 384, top-8, BF16 the sweep
(tools/ppu/qwen3_vl_moe/sweep_moe_tiles.py) gives 1.1-1.25x for M<=1024 and
2.4x for M>=4096, where Gems' default switches to its H20-tuned GEMM fast
path. Other shapes and M<16 / M>8192 keep Gems' own selection; the kernels
(fused_moe_kernel, moe_align, moe_sum) and their coverage are unchanged.
"""
import os
import sys

_installed = False

# Qwen3-VL-30B-A3B MoE weight shapes on one TP2 rank.
W1_SHAPE = (128, 768, 2048)
W2_SHAPE = (128, 2048, 384)


def _cfg(bm, bn, bk, warps=4, stages=3, group_m=1):
    return {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": group_m, "num_warps": warps, "num_stages": stages}


# (min_m, max_m) inclusive -> (gemm1, gemm2). BLOCK_SIZE_M is shared by the
# align step and both GEMMs, so each entry keeps one value for both.
TILES = (
    ((16, 191), (_cfg(16, 64, 128), _cfg(16, 128, 64))),
    ((192, 383), (_cfg(32, 64, 128), _cfg(32, 128, 64))),
    ((384, 767), (_cfg(32, 128, 64), _cfg(32, 128, 64))),
    ((768, 3071), (_cfg(64, 128, 64), _cfg(64, 128, 64))),
    ((3072, 8192), (_cfg(64, 128, 64), _cfg(64, 64, 64, warps=8))),
)


def lookup_tiles(m):
    for (lo, hi), tiles in TILES:
        if lo <= m <= hi:
            return tiles
    return None


def install_tuned_moe_tiles():
    global _installed
    if _installed:
        return
    import flag_gems  # noqa: F401

    fm = sys.modules["flag_gems.fused.fused_moe"]
    gems_lookup = fm.try_get_optimal_moe_config

    def lookup(w1_shape, w2_shape, top_k, dtype, M, E, block_shape=None,
               gemm_stage="gemm1", enable_gemm_fast_path=False,
               return_is_embedded=False):
        tiles = None
        if (dtype == "bf16" and block_shape is None and top_k == 8
                and tuple(w1_shape) == W1_SHAPE and tuple(w2_shape) == W2_SHAPE):
            tiles = lookup_tiles(M)
        if tiles is None:
            return gems_lookup(w1_shape, w2_shape, top_k, dtype, M, E,
                               block_shape=block_shape, gemm_stage=gemm_stage,
                               enable_gemm_fast_path=enable_gemm_fast_path,
                               return_is_embedded=return_is_embedded)
        cfg = dict(tiles[0] if gemm_stage == "gemm1" else tiles[1])
        # Not "embedded": keeps Gems' per-stage lookups (which land here again).
        return (cfg, False) if return_is_embedded else cfg

    fm.try_get_optimal_moe_config = lookup
    _installed = True


# --- hybrid: DeepGEMM grouped GEMM for full prefill chunks ------------------
# tools/ppu/qwen3_vl_moe/bench_moe_deepgemm.py, per layer vs the tuned Gems
# tiles above: M=256 0.83x, M=1024 1.35x, M=4096 1.82x, M=8192 1.84x. DeepGEMM
# picks a different kernel pair per M bucket (1024 / 1536-6144 / 8192), each an
# uncovered kernel type. Full chunked-prefill steps are exactly
# max_num_batched_tokens (8192) and are >90% of prefill MoE calls, so only
# those use DeepGEMM; the few remainder chunks stay on Gems (~0.4% slower).
# Decode stays on Gems (faster there, keeps fused_moe/moe_align/moe_sum covered).
DEEPGEMM_MIN_TOKENS = int(os.environ.get("VLLM_FL_QWEN3VL_DEEPGEMM_MIN", "8192"))


def hybrid_experts_class():
    from vllm_fl.ops.fused_moe.deepgemm_experts import DeepGemmExpertsFL
    from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL

    from ..minimax_m3.moe import disable_deepgemm_gemv

    class HybridQwen3VLExperts(TritonExpertsFL):
        # DeepGEMM's compact scratch (M*topk rows) also covers the Gems fast
        # path, which allocates its own caches and only writes ``output``.
        workspace_shapes = DeepGemmExpertsFL.workspace_shapes

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Its small-K GEMV is wrong on PPU; unreachable at M>=1024 but cheap.
            disable_deepgemm_gemv()

        def apply(self, output, hidden_states, *args, **kwargs):
            if hidden_states.shape[0] >= DEEPGEMM_MIN_TOKENS:
                return DeepGemmExpertsFL.apply(self, output, hidden_states, *args, **kwargs)
            return super().apply(output, hidden_states, *args, **kwargs)

    return HybridQwen3VLExperts
