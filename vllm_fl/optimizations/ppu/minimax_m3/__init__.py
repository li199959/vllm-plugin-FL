# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-M3 opt-in kernels for T-Head PPU.

VLLM_FL_MOE=hybrid selects DeepGEMM decode and FlagGems prefill experts.
VLLM_FL_M3_DENSE=hybrid selects split-K / FlagGems small-batch linears.
VLLM_FL_M3_GEMMA_NORM=flaggems selects FlagGems Gemma RMSNorm.
VLLM_FL_M3_FUSED_IMPL=triton selects fused Q/K norm, RoPE and KV insertion.
Any switch also installs shared shape-stable index / MoE alignment kernels.
See ../README.md for supported configurations and required exclusions."""

import os

VENDOR = "thead"
_registered = False


def _env(name):
    return os.environ.get(name, "").strip().lower()


def enabled() -> bool:
    return (
        _env("VLLM_FL_MOE") == "hybrid"
        or _env("VLLM_FL_M3_DENSE") == "hybrid"
        or _env("VLLM_FL_M3_GEMMA_NORM") == "flaggems"
        or _env("VLLM_FL_M3_FUSED_IMPL") == "triton"
    )


def register() -> None:
    """Install requested implementations after native schemas are registered."""
    global _registered
    if _registered:
        return
    from vllm.logger import init_logger

    from ..common.shape_stable import install_shape_stable_gems

    logger = init_logger(__name__)
    # Gems index/moe_align otherwise recompile for each new prefill size.
    install_shape_stable_gems()
    if _env("VLLM_FL_M3_DENSE") == "hybrid":
        from .linear import register_hybrid_linear

        register_hybrid_linear()
        logger.info_once("MiniMax-M3 decode linears use split-K Triton / tuned Gems")
    if _env("VLLM_FL_M3_GEMMA_NORM") == "flaggems":
        from .norm import register_gemma_norm

        register_gemma_norm()
        logger.info_once("MiniMax-M3 Gemma RMSNorm uses FlagGems")
    if _env("VLLM_FL_M3_FUSED_IMPL") == "triton":
        from .qknorm_rope import register_triton_qknorm_rope

        register_triton_qknorm_rope()
        logger.info_once("MiniMax-M3 Q/K norm, RoPE and KV insert use Triton")
    _registered = True


def moe_experts():
    if _env("VLLM_FL_MOE") != "hybrid":
        return None
    from vllm.logger import init_logger

    from .moe import HybridM3Experts

    init_logger(__name__).info_once(
        "Using HybridM3Experts (DeepGEMM decode without GEMV, Gems prefill)"
    )
    return HybridM3Experts
