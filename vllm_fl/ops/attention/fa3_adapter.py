# Copyright (c) 2025 BAAI. All rights reserved.
"""vLLM-signature adapter over the PPU-native FlashAttention-3 package.

The plugin's ``AttentionFLImpl`` calls a single ``flash_attn_varlen_func`` with
vLLM's calling convention (paged ``block_table``, ``out=``, ``seqused_k``,
``scheduler_metadata``, descales, ``return_softmax_lse``). The standalone
``flash_attn_interface`` package (``torch.ops.flash_attn_3``, PPU-native) does
NOT accept that signature directly:

* paged attention must go through ``flash_attn_with_kvcache(page_table=...)``;
* its varlen entry point has a different argument order and no ``out=`` /
  ``block_table`` / ``scheduler_metadata`` / ``return_softmax_lse``.

This module bridges the two. It is only imported when ``VLLM_FL_ATTN=fa3``
(see ``vllm_fl.utils.use_fa3_attention``); the default FlagGems path is
untouched.

Verified against ``fa3_microbench`` on PPU-ZW810E: head_dim=256, GQA 16/2,
packed varlen q ``(total_q, n_heads, head_dim)`` fed straight into
``flash_attn_with_kvcache`` (no reshape), numerics match SDPA (max_rel<0.005),
and paged block_size need NOT be a multiple of 256 on this build.
"""

import flash_attn_interface as _fa3


def _window(window_size):
    if window_size is None:
        return (-1, -1)
    return tuple(window_size)


def flash_attn_varlen_func(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    max_seqlen_q,
    max_seqlen_k=None,
    seqused_k=None,
    cu_seqlens_k=None,
    out=None,
    softmax_scale=None,
    causal=False,
    alibi_slopes=None,
    window_size=None,
    block_table=None,
    softcap=0.0,
    scheduler_metadata=None,
    fa_version=3,  # accepted and ignored; FA3 is the only backend here
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits=0,
    s_aux=None,
    return_softmax_lse=False,
):
    """Drop-in for the vLLM-signature ``flash_attn_varlen_func`` used by the FL
    attention impl, backed by the standalone FA3 kernels.

    Returns ``out`` (or ``(out, lse)`` when ``return_softmax_lse``); when an
    ``out`` buffer is supplied the result is copied into it and that buffer is
    returned, matching vLLM's in-place expectation.
    """
    assert alibi_slopes is None, "FA3 adapter does not support alibi_slopes"
    ws = _window(window_size)

    if block_table is not None:
        # Paged decode/prefill -> flash_attn_with_kvcache. vLLM passes packed
        # varlen q (total_q, n_heads, head_dim) + cu_seqlens_q; the PPU kernel
        # consumes it directly (verified in fa3_microbench).
        res = _fa3.flash_attn_with_kvcache(
            q=q,
            k_cache=k,
            v_cache=v,
            page_table=block_table,
            cache_seqlens=seqused_k,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=ws,
            softcap=softcap,
            num_splits=num_splits or 0,
            scheduler_metadata=scheduler_metadata,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            s_aux=s_aux,
            return_softmax_lse=return_softmax_lse,
        )
    else:
        # Non-paged prefill (encoder / DCP self-attention) -> varlen entry.
        assert cu_seqlens_k is not None, (
            "non-paged FA3 path requires cu_seqlens_k"
        )
        res = _fa3.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=ws,
            softcap=softcap,
            num_splits=num_splits or 1,
            s_aux=s_aux,
            return_attn_probs=return_softmax_lse,
        )

    if return_softmax_lse:
        if isinstance(res, (tuple, list)):
            attn_out, lse = res[0], res[1]
        else:  # kernel returned only the output tensor
            attn_out, lse = res, None
        if out is not None:
            out.copy_(attn_out)
            attn_out = out
        return attn_out, lse

    attn_out = res[0] if isinstance(res, (tuple, list)) else res
    if out is not None:
        out.copy_(attn_out)
        return out
    return attn_out


def get_scheduler_metadata(*args, **kwargs):
    """Pass-through to FA3 (signature identical to vLLM's). The FL metadata
    builder currently always uses ``scheduler_metadata=None``, so this is
    provided for completeness/parity only."""
    return _fa3.get_scheduler_metadata(*args, **kwargs)
