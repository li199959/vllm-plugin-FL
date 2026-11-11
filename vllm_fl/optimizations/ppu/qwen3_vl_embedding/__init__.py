# Copyright (c) 2026 BAAI. All rights reserved.
"""Qwen3-VL-Embedding-8B (Qwen3VLForConditionalGeneration, last-token pooling) on PPU.

The 8B dense LLM is GEMM bound, and the Gems mm/addmm/linear it gets by default
is 12-26% slower than cuBLAS for M >= 1024, so the serving config blacklists
them. Switches (opt-in; unset keeps the original configuration):
  VLLM_FL_Q3VLEMB_STABLE=1          shape-stable Gems index / copy_ (common/shape_stable).
                                    The last-token pooler gathers rows of the
                                    [num_tokens, 4096] hidden states with aten.index, and
                                    Gems bakes num_tokens into a new kernel per batch
                                    (10+ s p99 stalls at 1k input).
  VLLM_FL_Q3VLEMB_DENSE=hybrid      batches <= VLLM_FL_Q3VLEMB_GEMS_SMALL_M tokens run
                                    every LLM linear on a tuned Gems linear_kernel
                                    (near cuBLAS parity there); larger ones keep cuBLAS

Needed independently of these switches: the Gems flash-attention kernel returns
garbage/NaN for head_dim 72 at 96-256 tokens, and the ViT (head_dim 72, TORCH_SDPA)
reaches it for small images. Serve with
VLLM_FL_FLAGOS_BLACKLIST_APPEND=scaled_dot_product_attention,
_scaled_dot_product_flash_attention,_flash_attention_forward.
"""

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
        from ..common.shape_stable import install_shape_stable_copy, install_shape_stable_gems

        install_shape_stable_gems()
        install_shape_stable_copy()
    if _env("VLLM_FL_Q3VLEMB_DENSE") == "hybrid":
        from .linear import register_hybrid_linear

        register_hybrid_linear()
    _registered = True
