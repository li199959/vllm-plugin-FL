# Copyright (c) 2026 BAAI. All rights reserved.
"""BAAI bge-m3 (XLMRobertaModel, dense CLS embedding, pooling runner) on PPU.

Every serving batch has a different total token count, and the CLS pooler
gathers rows out of that [num_tokens, 1024] tensor with aten.index. Gems
generates an index kernel with that size baked in, so the original
configuration codegens/tunes a kernel on almost every batch (multi-second
stalls). Switches (all opt-in; unset keeps the original configuration):
  VLLM_FL_BGEM3_STABLE=1        shape-stable Gems index / copy_ (common/shape_stable)
  VLLM_FL_BGEM3_DENSE=hybrid    linears in VLLM_FL_BGEM3_GEMS_LINEARS (qkv,o,fc1,fc2;
                                default o) on a tuned Gems linear_kernel, plus every
                                projection for batches <= VLLM_FL_BGEM3_GEMS_SMALL_M
                                tokens (default 0 = off); the rest stays on cuBLAS, so
                                meant for configs that blacklist Gems mm/addmm/linear
  VLLM_FL_BGEM3_ATTN=fa3        encoder-only attention on the PPU FA3 wheel instead of
                                vLLM's Triton context_attention_fwd (3.3x faster)

Needed independently of these switches: the Gems ``broadcast_to`` override does
a pinned H2D copy, which is illegal during CUDA graph capture, and XLM-R's
biased linears reach it through addmm. Serve with
VLLM_FL_FLAGOS_BLACKLIST_APPEND=broadcast_to (aten's broadcast_to is a view).
"""

import os

VENDOR = "thead"
_registered = False


def _env(name):
    return os.environ.get(name, "").strip().lower()


def _stable():
    return _env("VLLM_FL_BGEM3_STABLE") in ("1", "true", "on")


def enabled() -> bool:
    return (_stable() or _env("VLLM_FL_BGEM3_DENSE") == "hybrid"
            or _env("VLLM_FL_BGEM3_ATTN") == "fa3")


def register() -> None:
    global _registered
    if _registered:
        return
    if _stable():
        from ..common.shape_stable import install_shape_stable_copy, install_shape_stable_gems

        install_shape_stable_gems()
        install_shape_stable_copy()
    if _env("VLLM_FL_BGEM3_DENSE") == "hybrid":
        from .linear import register_hybrid_linear

        register_hybrid_linear()
    if _env("VLLM_FL_BGEM3_ATTN") == "fa3":
        from .attention import install_fa3_encoder_attention

        install_fa3_encoder_attention()
    _registered = True
