# Copyright (c) 2026 BAAI. All rights reserved.
"""OpenAI Whisper (WhisperForConditionalGeneration, /v1/audio/transcriptions) on PPU.

vLLM runs every engine step that carries encoder inputs fully eager: the 32-layer
audio encoder is not compiled at all, and the decoder skips its compiled graph
(``skip_compiled``) whenever a new request is scheduled. With FlagGems enabled each
eager op pays Gems' Python dispatch (~150-550 us per add/layer_norm/gelu/clamp/sdpa
vs ~30 us for aten), so at 64 concurrency the engine core is CPU bound (GPU busy
42%) and throughput is ~64% of the attention-only configuration.

Switches (all opt-in; unset keeps vLLM's behaviour):
  VLLM_FL_WHISPER_COMPILE_ENCODER=1  run the encoder layer stack + final norm through
                                     torch.compile (inductor, batch dim dynamic), so
                                     pointwise/norm ops become fused Triton kernels
  VLLM_FL_WHISPER_ENCODER_GRAPH_MAX=N  with COMPILE_ENCODER: replay a CUDA graph of the
                                     compiled stack for batches of <= N audios (captured
                                     on first use per batch size); default 0 = off
  VLLM_FL_WHISPER_COMPILE_PREFILL=1  steps with encoder inputs write the cross-attention
                                     K/V eagerly, then run the compiled decoder graph
                                     (same graph as decode steps) instead of eager
  VLLM_FL_WHISPER_CROSS_SLOTS=1      batched cross-attention slot mapping (cross_slots.py):
                                     one gather per step instead of 6 tiny ops per new
                                     request in the attention-metadata build
  VLLM_FL_WHISPER_STABLE=1           shape-stable Gems index / copy_ (common/shape_stable);
                                     the cross-attention slot mapping gathers with a new
                                     size every step (~765 us/call with per-shape codegen)

Only WhisperModel instances are patched; other models that reuse WhisperEncoder
(Qwen2-Audio, Kimi-Audio, ...) keep the original code.
"""

import os

VENDOR = "thead"
_registered = False


def _on(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on")


def enabled() -> bool:
    return (_on("VLLM_FL_WHISPER_COMPILE_ENCODER") or _on("VLLM_FL_WHISPER_COMPILE_PREFILL")
            or _on("VLLM_FL_WHISPER_STABLE") or _on("VLLM_FL_WHISPER_CROSS_SLOTS"))


def register() -> None:
    global _registered
    if _registered:
        return
    if _on("VLLM_FL_WHISPER_STABLE"):
        from ..common.shape_stable import install_shape_stable_copy, install_shape_stable_gems

        install_shape_stable_gems()
        install_shape_stable_copy()
    if _on("VLLM_FL_WHISPER_CROSS_SLOTS"):
        from .cross_slots import install as install_cross_slots

        install_cross_slots()
    encoder, prefill = _on("VLLM_FL_WHISPER_COMPILE_ENCODER"), _on("VLLM_FL_WHISPER_COMPILE_PREFILL")
    if encoder or prefill:
        from .compile import install

        install(encoder=encoder, prefill=prefill)
    _registered = True
