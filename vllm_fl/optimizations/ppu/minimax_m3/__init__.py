# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-M3 on PPU: FlagGems/Triton coverage with <=5% throughput loss.

Switches (all opt-in; unset keeps the original configuration):
  VLLM_FL_MOE=hybrid             decode (<=128 tokens) on DeepGEMM grouped GEMM with
                                 its GEMV path disabled; prefill on tuned Gems
  VLLM_FL_M3_DENSE=hybrid        per-M split-K Triton / tuned Gems for decode linears
                                 (BF16 M<=128, FP32 router M<=512)
  VLLM_FL_M3_GEMMA_NORM=flaggems Gems Gemma RMSNorm / fused add RMSNorm
  VLLM_FL_M3_FUSED_IMPL=triton   Triton Q/K norm + RoPE + KV insert
Any switch also installs the shape-stable Gems index / moe_align kernels.
Keep `index` in VLLM_FL_FLAGOS_BLACKLIST (see README: ~1 GiB less KV cache).
See README.md in this directory for measurements and the launch recipe.
"""

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

    from .shape_stable import install_shape_stable_gems

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
