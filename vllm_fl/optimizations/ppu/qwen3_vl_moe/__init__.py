# Copyright (c) 2026 BAAI. All rights reserved.
"""Qwen3-VL-30B-A3B opt-in kernels for T-Head PPU.

VLLM_FL_QWEN3VL_STABLE=1 enables shape-stable index / copy / MoE alignment.
VLLM_FL_QWEN3VL_ATTN=tuned tunes Triton decode; hybrid uses FA3 for prefill;
fa3 also routes eligible decode to the vendor FA3 wheel.
VLLM_FL_QWEN3VL_MOE=tuned selects FlagGems tiles; hybrid uses DeepGEMM
above VLLM_FL_QWEN3VL_DEEPGEMM_MIN (default 8192 tokens).
VLLM_FL_QWEN3VL_DENSE=tuned selects supported TP2 decode projections.
See ../README.md for configuration and dependencies."""

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
        from ..common.shape_stable import (
            install_shape_stable_copy,
            install_shape_stable_gems,
        )

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
