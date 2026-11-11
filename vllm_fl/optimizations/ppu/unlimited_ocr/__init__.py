# Copyright (c) 2026 BAAI. All rights reserved.
"""baidu/Unlimited-OCR (UnlimitedOCRForCausalLM) on PPU.

Model support itself (config, R-SWA attention) lives in vllm_fl/models/unlimited_ocr.
Every OCR request has a different token count (image tiles + generated text), so
the original configuration keeps recompiling Gems ``index`` and
``moe_align_block_size_stage1`` during serving. Switches (all opt-in; unset keeps
the original configuration):
  VLLM_FL_UOCR_STABLE=1   shape-stable Gems index / moe_align stage1 / copy_
                          (common/shape_stable)
"""

import os

VENDOR = "thead"
_registered = False


def _stable():
    return os.environ.get("VLLM_FL_UOCR_STABLE", "").strip().lower() in ("1", "true", "on")


def enabled() -> bool:
    return _stable()


def register() -> None:
    global _registered
    if _registered:
        return
    if _stable():
        from ..common.shape_stable import install_shape_stable_copy, install_shape_stable_gems

        install_shape_stable_gems()
        install_shape_stable_copy()
    _registered = True
