# Copyright (c) 2026 BAAI. All rights reserved.
"""Whisper opt-in encoder and decoder optimizations for T-Head PPU.

VLLM_FL_WHISPER_COMPILE_ENCODER=1 compiles the encoder layer stack.
VLLM_FL_WHISPER_ENCODER_GRAPH_MAX=N also captures batches of at most N audios
(default 0, disabled). VLLM_FL_WHISPER_COMPILE_PREFILL=1 writes cross K/V
eagerly and runs the compiled decoder with the populated cache.
VLLM_FL_WHISPER_CROSS_SLOTS=1 batches the shared cross-attention slot builder.
VLLM_FL_WHISPER_STABLE=1 installs shape-stable index / copy kernels.
Compilation patches apply only to WhisperModel. See ../README.md."""

import os

VENDOR = "thead"
_registered = False


def _on(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on")


def enabled() -> bool:
    return (
        _on("VLLM_FL_WHISPER_COMPILE_ENCODER")
        or _on("VLLM_FL_WHISPER_COMPILE_PREFILL")
        or _on("VLLM_FL_WHISPER_STABLE")
        or _on("VLLM_FL_WHISPER_CROSS_SLOTS")
    )


def register() -> None:
    global _registered
    if _registered:
        return
    if _on("VLLM_FL_WHISPER_STABLE"):
        from ..common.shape_stable import (
            install_shape_stable_copy,
            install_shape_stable_gems,
        )

        install_shape_stable_gems()
        install_shape_stable_copy()
    if _on("VLLM_FL_WHISPER_CROSS_SLOTS"):
        from .cross_slots import install as install_cross_slots

        install_cross_slots()
    encoder, prefill = (
        _on("VLLM_FL_WHISPER_COMPILE_ENCODER"),
        _on("VLLM_FL_WHISPER_COMPILE_PREFILL"),
    )
    if encoder or prefill:
        from .compile import install

        install(encoder=encoder, prefill=prefill)
    _registered = True
