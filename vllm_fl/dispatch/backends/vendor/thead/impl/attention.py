# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Thead (T-Head / PPU) FlashAttention backend.
#
# This module provides a custom attention backend for PPU accelerators that
# uses the flash_attn_3 wheel (torch.ops.flash_attn_3.fwd) directly.
#
# At module load time we:
#   1. Import flash_attn_3._C to register FA3 custom ops.
#   2. Provide a custom flash_attn_varlen_func that calls the wheel's fwd
#      with the correct arg signature (35 args), bridging differences between
#      v0.20.2's FA3 branch (which passes cp_* extras and skips attention_chunk)
#      and the wheel's expected signature.
#   3. Inject the needed functions into the flash_attn module namespace so that
#      the inherited FlashAttentionImpl.forward() can resolve them.
#   4. Provide a pure-PyTorch reshape_and_cache_flash for PPU (no _C.abi3.so).
#   5. Handle PPU-specific requirements:
#      - When cu_seqlens_k is None (paged attention), max_seqlen_k must be 1.
#      - FA3 kernel uses max_seqlen_k to select tile size (Aone#75639039).

from __future__ import annotations

import os
from typing import ClassVar

import torch

# ---------------------------------------------------------------------------
# Step 1 — load the flash_attn_3 wheel
# ---------------------------------------------------------------------------
import flash_attn_3._C  # noqa: F401 — registers torch.ops.flash_attn_3


# ---------------------------------------------------------------------------
# Step 2 — provide a custom flash_attn_varlen_func for PPU
# ---------------------------------------------------------------------------
# v0.20.2's FA3 branch calls torch.ops._vllm_fa3_C.fwd() with 37 args:
#   ... softcap, True(=rotary_interleaved), scheduler_metadata, num_splits,
#       None(=pack_gqa), 0(=sm_margin), s_aux,
#       cp_world_size, cp_rank, cp_tot_seqused_k     <-- extras
# BUT the flash_attn_3 wheel expects 35 args:
#   ... window_size_right, attention_chunk, softcap, is_rotary_interleaved,
#       scheduler_metadata, num_splits, pack_gqa, sm_margin, s_aux
#
# So we provide our own varlen wrapper that calls the wheel directly.


def _thead_flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: list[int] | None = None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    # Version selector (ignored — we always use FA3)
    fa_version: int = 3,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
):
    """Custom flash_attn_varlen_func for PPU using the flash_attn_3 wheel.

    Accepts the same signature as vLLM's flash_attn_varlen_func (including
    the extra cp_* args), but calls torch.ops.flash_attn_3.fwd with the
    correct 35-argument signature.
    """
    del fa_version, cp_world_size, cp_rank, cp_tot_seqused_k  # unused
    del dropout_p, deterministic, return_attn_probs  # unused in FA3

    assert alibi_slopes is None, "Alibi is not supported in FA3"

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    real_window_size: tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])

    # PPU Note Aone#75639039:
    # PPU FA3 uses max_seqlen_k to choose tile size.
    # In paged attention cu_seqlens_k is None, force max_seqlen_k = 1.
    if cu_seqlens_k is None:
        max_seqlen_k = 1

    # The PPU wheel does not expose FA3's AOT scheduler metadata API. The
    # num_splits=0 runtime heuristic is not CUDA Graph compatible, so use a
    # fixed launch shape when no precomputed schedule is available.
    if scheduler_metadata is None and num_splits == 0:
        num_splits = 1

    # A fixed split count is graph-safe even without FA3 AOT metadata. On
    # Qwen3.5 TP2, batch-64 decode with a 4k-5k KV length benefits from five
    # splits; keep this opt-in because the best value depends on the workload.
    decode_splits = os.getenv("VLLM_FL_FA_DECODE_SPLITS", "")
    if (
        decode_splits
        and scheduler_metadata is None
        and cu_seqlens_k is None
        and max_seqlen_q == 1
        and os.getenv("VLLM_BATCH_INVARIANT", "0") != "1"
    ):
        num_splits = int(decode_splits)
        if num_splits not in (1, 2, 4, 5, 8, 16, 32):
            raise ValueError("VLLM_FL_FA_DECODE_SPLITS must be 1, 2, 4, 5, 8, 16, or 32")

    out, softmax_lse, _, _ = torch.ops.flash_attn_3.fwd(
        q, k, v,
        None, None,  # k_new, v_new
        q_v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        None,
        seqused_k,  # seqused_q, seqused_k
        max_seqlen_q,
        max_seqlen_k,
        block_table,
        None,  # kv_batch_idx
        None,  # leftpad_k
        None, None, None,  # rotary_cos, rotary_sin, seqlens_rotary
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        real_window_size[0],
        real_window_size[1],
        0,      # attention_chunk
        softcap,
        True,   # is_rotary_interleaved
        scheduler_metadata,
        num_splits,
        None,   # pack_gqa
        0,      # sm_margin
        s_aux,
    )

    return (out, softmax_lse) if return_softmax_lse else out


# ---------------------------------------------------------------------------
# Step 2b — inject into flash_attn module namespace
# ---------------------------------------------------------------------------
import vllm.v1.attention.backends.flash_attn as _flash_attn_mod
from vllm import vllm_flash_attn as _vfa

# Prefer the native vLLM extension when available; otherwise use the
# standalone PPU wheel bridge above.
try:
    from vllm.vllm_flash_attn import flash_attn_interface as _native_fa
    if getattr(_native_fa, "FA3_AVAILABLE", False):
        _NATIVE_FA_VERSION: int | None = 3
    elif getattr(_native_fa, "FA2_AVAILABLE", False):
        _NATIVE_FA_VERSION = 2
    else:
        _NATIVE_FA_VERSION = None
except Exception:
    _native_fa = None
    _NATIVE_FA_VERSION = None

if _NATIVE_FA_VERSION is not None:
    # Attention read via vLLM-native FA (stable). KV-cache write still uses the
    # pure-torch PPU path below (native reshape_and_cache is not built on PPU).
    _flash_attn_mod.flash_attn_varlen_func = _native_fa.flash_attn_varlen_func
    _flash_attn_mod.get_scheduler_metadata = _native_fa.get_scheduler_metadata
else:
    _flash_attn_mod.flash_attn_varlen_func = _thead_flash_attn_varlen_func
    _flash_attn_mod.get_scheduler_metadata = _vfa.get_scheduler_metadata

# ---------------------------------------------------------------------------
# Step 2c — pure-PyTorch reshape_and_cache_flash for PPU
# ---------------------------------------------------------------------------
# The original is a CUDA custom op from _C.abi3.so which is not available
# on the remote.  We provide a pure-PyTorch indexed-copy version.


def reshape_and_cache_flash_thead(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """GPU-only KV cache write for PPU, compatible with CUDA graph capture.

    The original CUDA custom op (``_C_cache_ops.reshape_and_cache_flash``)
    is not available on the remote.  This pure-PyTorch equivalent avoids
    *any* CPU-GPU synchronisation or data-dependent shape changes so that
    it can run inside a CUDA graph capture region.

    Padding tokens (``slot_mapping == -1``) are handled by zeroing their
    key/value before writing to a safe slot, rather than skipping them with
    a conditional — the latter would require a CPU sync (``.any()``) and
    produce a data-dependent tensor shape. Slot 0 is vLLM's reserved null block
    (``BlockPool.null_block``, block_id 0), so those writes never touch a real
    token's entry.

    The PPU wheel may read the final partial 16-slot tile beyond
    seqused_k. Clear up to 15 following slots inside each block so stale
    NaN/Inf values from recycled blocks cannot corrupt attention output.
    Write real K/V last to preserve tokens that overlap these tails.
    All operations have static shapes and require no host synchronization.
    This mitigates missing partial-tile masking in the vendor kernel.
    """
    del kv_cache_dtype, k_scale, v_scale  # unused in pure-torch path

    valid_mask_gpu = (slot_mapping >= 0).to(key.dtype).view(-1, 1, 1)
    masked_key = key * valid_mask_gpu
    masked_value = value * valid_mask_gpu

    safe_slots = slot_mapping.clamp(min=0)  # -1 -> 0 (reserved null block)

    # Convert flat slot indices to (block, token_within_block) coordinates.
    # key_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
    block_size = key_cache.shape[1]
    block_indices = safe_slots // block_size
    token_in_block = safe_slots % block_size

    # --- tail zeroing: define the slots the kernel may over-read ---
    _TILE = 16
    offsets = torch.arange(1, _TILE, device=safe_slots.device)       # 1..15
    tail_tok = token_in_block.unsqueeze(1) + offsets.unsqueeze(0)    # [N, 15]
    # stay inside this slot's own block: another block may belong to a
    # different sequence, and clamping keeps every index in range.
    tail_tok = tail_tok.clamp(max=block_size - 1)
    tail_blk = block_indices.unsqueeze(1).expand_as(tail_tok)
    tail_blk_flat = tail_blk.reshape(-1)
    tail_tok_flat = tail_tok.reshape(-1)
    # Materialise an explicit zero tensor of the exact destination shape rather
    # than assigning the scalar 0: a scalar RHS routes through FlagGems'
    # index_put_ broadcast path, which faults on this shape
    # (flag_gems/ops/index_put.py:342, device-side assert).
    zeros_tail = key_cache.new_zeros(
        (tail_blk_flat.numel(), key_cache.shape[2], key_cache.shape[3])
    )
    key_cache[tail_blk_flat, tail_tok_flat] = zeros_tail
    value_cache[tail_blk_flat, tail_tok_flat] = zeros_tail

    # real K/V last, so it wins over any zero written above
    key_cache[block_indices, token_in_block] = masked_key
    value_cache[block_indices, token_in_block] = masked_value


_flash_attn_mod.reshape_and_cache_flash = reshape_and_cache_flash_thead

# ---------------------------------------------------------------------------
# Step 3 — custom backend & impl
# ---------------------------------------------------------------------------

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    flash_attn_supports_sinks,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)


class TheadFlashAttentionImpl(FlashAttentionImpl):
    """FlashAttention implementation for PPU that uses FA3 (flash_attn_3 wheel).

    The only difference from FlashAttentionImpl:
    - vllm_flash_attn_version is forced to 3 (FA3) regardless of CC.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        # FA version: use the vLLM-native FA version when its extension is built
        # (2 for _vllm_fa2_C, 3 for _vllm_fa3_C); otherwise fall back to 3 for
        # the flash_attn_3 wheel bridge.
        self.vllm_flash_attn_version = (
            _NATIVE_FA_VERSION if _NATIVE_FA_VERSION is not None else 3
        )


class TheadFlashAttentionBackend(FlashAttentionBackend):
    """FlashAttention backend for PPU that delegates to TheadFlashAttentionImpl."""

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[TheadFlashAttentionImpl]:
        return TheadFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return FlashAttentionMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # PPU CC = 8.0
        return capability >= DeviceCapability(8, 0) and capability < DeviceCapability(9, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink:
            return "sink not supported on PPU (CC < 9.0)"
        if use_mla:
            return "MLA not supported in thead flash attention backend"
        return None
