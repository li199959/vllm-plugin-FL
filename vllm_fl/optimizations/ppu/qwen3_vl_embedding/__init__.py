# Copyright (c) 2026 BAAI. All rights reserved.
"""Qwen3-VL-Embedding-8B opt-in kernels for PPU last-token pooling.

VLLM_FL_Q3VLEMB_STABLE=1 enables shape-stable index and copy kernels.
VLLM_FL_Q3VLEMB_DENSE=hybrid routes supported LLM projections with at most
VLLM_FL_Q3VLEMB_GEMS_SMALL_M tokens through explicit FlagGems tiles.
The threshold defaults to zero (disabled); larger batches keep the native path.
See ../README.md for SDPA exclusions and the required request template."""

import os

VENDOR = "thead"
_registered = False


def _env(name):
    return os.environ.get(name, "").strip().lower()


def _stable():
    return _env("VLLM_FL_Q3VLEMB_STABLE") in ("1", "true", "on")


def enabled() -> bool:
    return _stable() or _env("VLLM_FL_Q3VLEMB_DENSE") == "hybrid"


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
    if _env("VLLM_FL_Q3VLEMB_DENSE") == "hybrid":
        from .linear import register_hybrid_linear

        register_hybrid_linear()
    _registered = True
