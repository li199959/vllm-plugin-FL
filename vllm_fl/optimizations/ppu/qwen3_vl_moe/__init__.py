# Copyright (c) 2026 BAAI. All rights reserved.
"""Qwen3-VL-30B-A3B (Qwen3VLMoeForConditionalGeneration) on PPU.

The original configuration already routes most ops through FlagGems/Triton
(combined coverage 39/44); what it loses is throughput to Triton compiles
during serving and to untuned launch configs. Switches (all opt-in; unset
keeps the original configuration):
  VLLM_FL_QWEN3VL_STABLE=1      shape-stable Gems index / moe_align stage1 / copy_
  VLLM_FL_QWEN3VL_ATTN=tuned    PPU launch params for Triton unified attention (decode)
  VLLM_FL_QWEN3VL_ATTN=hybrid   tuned + steps containing prefill on the PPU FA3 wheel
  VLLM_FL_QWEN3VL_ATTN=fa3      every eligible step (decode too) on PPU FA3
  VLLM_FL_QWEN3VL_MOE=tuned     PPU tiles for Gems fused MoE
  VLLM_FL_QWEN3VL_MOE=hybrid    DeepGEMM grouped GEMM for full 8192-token prefill chunks,
                                tuned Gems otherwise (VLLM_FL_QWEN3VL_DEEPGEMM_MIN)
  VLLM_FL_QWEN3VL_DENSE=tuned   decode-size qkv/o_proj/router on Gems linear_kernel tiles
See README.md in this directory for measurements and the launch recipe.
"""

import os

VENDOR = "thead"
_registered = False


def _env(name):
    return os.environ.get(name, "").strip().lower()


def _stable():
    return _env("VLLM_FL_QWEN3VL_STABLE") in ("1", "true", "on")


def enabled() -> bool:
    return (
        _stable()
        or _env("VLLM_FL_QWEN3VL_ATTN") in ("tuned", "hybrid", "fa3")
        or _env("VLLM_FL_QWEN3VL_MOE") in ("tuned", "hybrid")
        or _env("VLLM_FL_QWEN3VL_DENSE") == "tuned"
    )


def register() -> None:
    global _registered
    if _registered:
        return
    if _stable():
        from ..common.shape_stable import install_shape_stable_copy, install_shape_stable_gems

        # Chunked prefill gives a new token count almost every step; without
        # these each one recompiles Gems index / moe_align / copy_.
        install_shape_stable_gems()
        install_shape_stable_copy()
    attn = _env("VLLM_FL_QWEN3VL_ATTN")
    if attn in ("tuned", "hybrid"):
        from .attention import install_tuned_attention

        install_tuned_attention()
    if attn in ("hybrid", "fa3"):
        from .attention import install_hybrid_attention

        install_hybrid_attention(decode=attn == "fa3")
    if _env("VLLM_FL_QWEN3VL_MOE") in ("tuned", "hybrid"):
        from .moe import install_tuned_moe_tiles

        install_tuned_moe_tiles()
    if _env("VLLM_FL_QWEN3VL_DENSE") == "tuned":
        from .linear import register_tuned_linear

        register_tuned_linear()
    _registered = True


def moe_experts():
    if _env("VLLM_FL_QWEN3VL_MOE") != "hybrid":
        return None
    from .moe import hybrid_experts_class

    return hybrid_experts_class()
