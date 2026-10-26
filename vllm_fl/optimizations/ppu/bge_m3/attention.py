# Copyright (c) 2026 BAAI. All rights reserved.
"""bge-m3 encoder-only attention on the PPU FA3 wheel (VLLM_FL_BGEM3_ATTN=fa3).

vLLM's Triton backend runs ENCODER_ONLY attention through
triton_prefill_attention.context_attention_fwd. On PPU that kernel is 3.2-3.4x
slower than FA3 varlen (non-causal, 16 heads x 64, fp16: 10.3 vs 3.1 ms per layer
at one 8192-token sequence, 1.73 vs 0.54 ms at 8 x 1026), and attention is most of
the GPU time for long inputs. Outputs are identical (max abs diff 0). Calls that
FA3 can't take (sliding window, other dtypes, quantized KV) keep the Triton path.
"""

import os

import torch

_installed = False
# Batches below this many tokens keep Triton. FA3 switches to a second tile
# variant (64x128 instead of 128x64) under 2048 tokens; those are tail batches
# with little GPU time, and Triton costs 13-150 us more there.
_min_tokens = 2048


def min_tokens_from_env() -> int:
    raw = os.environ.get("VLLM_FL_BGEM3_FA3_MIN_TOKENS", "")
    return int(raw) if raw.strip() else 2048


def _fa3_eligible(impl, query, key):
    return (
        query.shape[0] >= _min_tokens
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == query.dtype
        and tuple(impl.sliding_window) in ((-1, -1), (None, None))
        and query.shape[-1] in (64, 96, 128, 192, 256)
    )


def fa3_encoder_attention(query, key, value, output, cu_seqlens, max_seqlen, scale):
    torch.ops.flash_attn_3.fwd(
        query, key, value, out=output, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, softmax_scale=scale,
        is_causal=False, num_splits=1)
    return output


def install_fa3_encoder_attention():
    global _installed, _min_tokens
    if _installed:
        return
    _min_tokens = min_tokens_from_env()
    import flash_attn_3._C  # noqa: F401  (registers torch.ops.flash_attn_3)
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    triton_encoder = TritonAttentionImpl._forward_encoder_attention

    def _forward_encoder_attention(self, query, key, value, output, attn_metadata, layer):
        if not _fa3_eligible(self, query, key):
            return triton_encoder(self, query, key, value, output, attn_metadata, layer)
        return fa3_encoder_attention(query, key, value, output, attn_metadata.query_start_loc,
                                     attn_metadata.max_query_len, self.scale)

    _forward_encoder_attention.triton = triton_encoder
    TritonAttentionImpl._forward_encoder_attention = _forward_encoder_attention
    _installed = True
