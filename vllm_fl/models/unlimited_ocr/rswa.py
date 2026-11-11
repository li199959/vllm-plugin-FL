# Copyright (c) 2026 BAAI. All rights reserved.
"""Reference Sliding Window Attention (R-SWA) for Unlimited-OCR on vLLM v0.24.

Unlimited-OCR is trained so that prompt/image tokens stay globally visible while
each *generated* token additionally sees only the last ``rswa_window`` tokens
(the HF reference keeps a ring buffer of that size after the prefill KV). Upstream
vLLM (#46564, #47102) adds this as an attention-metadata field; v0.24 has no such
field, so here:

- the model runner writes each request's prompt length into a persistent GPU
  buffer every step (``update_prefix_lens``), outside any CUDA graph, so captured
  graphs read the current values;
- ``triton_attn.unified_attention`` is wrapped to call the vendored kernel
  (``rswa_unified_attention``) with that buffer and the window.

Prefill tokens lie inside the prompt, so for them the mask equals plain causal;
only decode steps are affected. The KV cache is not trimmed (upstream's
``RSWAManager`` only saves memory).
"""

import torch

_window: int | None = None
_prefix_lens: torch.Tensor | None = None


def enabled() -> bool:
    return _window is not None


def install(window: int, max_num_seqs: int, device: torch.device) -> None:
    global _window, _prefix_lens
    if _window is not None:
        assert _window == window, (_window, window)
        return
    _prefix_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
    _window = window
    _install_attention_hook()


def update_prefix_lens(num_prompt_tokens_cpu: torch.Tensor) -> None:
    """Copy per-request prompt lengths (CPU, int32) into the persistent buffer."""
    if _prefix_lens is None:
        return
    n = min(num_prompt_tokens_cpu.shape[0], _prefix_lens.shape[0])
    _prefix_lens[:n].copy_(num_prompt_tokens_cpu[:n], non_blocking=True)


def _install_attention_hook() -> None:
    import vllm.v1.attention.backends.triton_attn as backend

    from .rswa_unified_attention import unified_attention as rswa_unified_attention

    original = backend.unified_attention
    if getattr(original, "rswa", False):
        return

    def unified_attention(**kw):
        num_seqs = kw["seqused_k"].shape[0]
        assert num_seqs <= _prefix_lens.shape[0], (num_seqs, _prefix_lens.shape[0])
        return rswa_unified_attention(
            **kw, rswa_prefix_lens=_prefix_lens[:num_seqs], rswa_window=_window
        )

    unified_attention.rswa = True
    unified_attention.original = original
    backend.unified_attention = unified_attention
