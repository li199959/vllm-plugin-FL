# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-M3 on PPU: FlagGems/Triton coverage with <=5% throughput loss.

Switches (all opt-in; unset keeps the original configuration):
  VLLM_FL_M3_GEMMA_NORM=flaggems Gems Gemma RMSNorm / fused add RMSNorm
  VLLM_FL_M3_FUSED_IMPL=triton   Triton Q/K norm + RoPE + KV insert
Any switch also installs the shape-stable Gems index / moe_align kernels.
"""

import os

VENDOR = "thead"
_registered = False


def _env(name):
    return os.environ.get(name, "").strip().lower()


def enabled() -> bool:
    return (
        _env("VLLM_FL_M3_GEMMA_NORM") == "flaggems"
        or _env("VLLM_FL_M3_FUSED_IMPL") == "triton"
    )


def register() -> None:
    """Install requested implementations after native schemas are registered."""
    global _registered
    if _registered:
        return
    from vllm.logger import init_logger

    from .shape_stable import install_shape_stable_gems

    logger = init_logger(__name__)
    # Gems index/moe_align otherwise recompile for each new prefill size.
    install_shape_stable_gems()
    if _env("VLLM_FL_M3_GEMMA_NORM") == "flaggems":
        from .norm import register_gemma_norm

        register_gemma_norm()
        logger.info_once("MiniMax-M3 Gemma RMSNorm uses FlagGems")
    if _env("VLLM_FL_M3_FUSED_IMPL") == "triton":
        from .qknorm_rope import register_triton_qknorm_rope

        register_triton_qknorm_rope()
        logger.info_once("MiniMax-M3 Q/K norm, RoPE and KV insert use Triton")
    _registered = True
