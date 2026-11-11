# Copyright (c) 2026 BAAI. All rights reserved.
"""BGE-M3 (XLMRobertaModel, dense CLS pooling) opt-in kernels for PPU.

VLLM_FL_BGEM3_STABLE=1 enables shape-stable index and copy kernels.
VLLM_FL_BGEM3_DENSE=hybrid uses FlagGems for selected projections and
small batches; VLLM_FL_BGEM3_GEMS_LINEARS and VLLM_FL_BGEM3_GEMS_SMALL_M
control selection. Other projections retain the configured native path.
VLLM_FL_BGEM3_ATTN=fa3 selects eligible encoder attention on the PPU FA3 wheel.
See ../README.md for required exclusions and configuration."""

import os

VENDOR = "thead"
_registered = False


def _env(name):
    return os.environ.get(name, "").strip().lower()


def _stable():
    return _env("VLLM_FL_BGEM3_STABLE") in ("1", "true", "on")


def enabled() -> bool:
    return (
        _stable()
        or _env("VLLM_FL_BGEM3_DENSE") == "hybrid"
        or _env("VLLM_FL_BGEM3_ATTN") == "fa3"
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

        install_shape_stable_gems()
        install_shape_stable_copy()
    if _env("VLLM_FL_BGEM3_DENSE") == "hybrid":
        from .linear import register_hybrid_linear

        register_hybrid_linear()
    if _env("VLLM_FL_BGEM3_ATTN") == "fa3":
        from .attention import install_fa3_encoder_attention

        install_fa3_encoder_attention()
    _registered = True
