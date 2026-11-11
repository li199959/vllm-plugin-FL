# Copyright (c) 2026 BAAI. All rights reserved.
"""Pure-torch fallback for ``_C::fused_minimax_m3_qknorm_rope_kv_insert``.

Mirrors ``csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu``
from upstream vLLM so MiniMax-M3 runs on builds without the native extension:

- Gemma RMSNorm per head (``x * rsqrt(mean(x^2)+eps) * (1 + w)``), computed in
  fp32 with a single rounding back to the qkv dtype at the store (the CUDA
  kernel keeps fp32 registers from norm through RoPE).
- Partial NeoX RoPE on the leading ``rotary_dim`` dims; the tail passes
  through. ``cos_sin_cache`` is ``[max_pos, rotary_dim]`` (cos | sin).
- ``qkv`` packs ``[q | k | v]`` (dense) or ``[q | k | v | index_q | index_k]``
  (sparse). V is never normed/roped.
- ``q_out`` / ``index_q_out`` (contiguous ``[N, heads*128]``) receive the
  normed/roped heads *instead of* the in-place update, matching the kernel's
  redirected store. The fp8 index path stores saturating e4m3 (±448) bytes.
- Cache inserts (sparse serving): K (post norm+RoPE) and V (raw) scatter into
  the paged ``kv_cache`` (logical ``[nb, 2, bs, nkv, 128]``, any strides with a
  contiguous head_dim) via ``slot_mapping``; the normed/roped index_k scatters
  into the flat ``index_cache [nb*bs, 128]`` via ``index_slot_mapping``
  (falling back to ``slot_mapping``). Slots < 0 are skipped. ``kv_cache_dtype``
  ``fp8_*`` writes scaled-convert (scale 1.0) values into uint8 storage.
"""

import torch

_HEAD_DIM = 128
_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0

SCHEMA = (
    "fused_minimax_m3_qknorm_rope_kv_insert("
    "Tensor! qkv, Tensor q_norm_weight, Tensor k_norm_weight, "
    "Tensor cos_sin_cache, Tensor positions, int num_heads, "
    "int num_kv_heads, int rotary_dim, float eps, "
    "Tensor? index_q_norm_weight, Tensor? index_k_norm_weight, "
    "int num_index_heads, "
    "Tensor? slot_mapping, Tensor? index_slot_mapping, "
    "Tensor!? kv_cache, Tensor!? index_cache, "
    "int block_size, Tensor!? q_out, Tensor!? index_q_out, "
    "str kv_cache_dtype) -> ()"
)


def _norm_rope(
    x32: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Gemma RMSNorm + partial NeoX RoPE, all in fp32. x32: [N, H, 128]."""
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    xn = x32 * torch.rsqrt(var + eps) * (1.0 + weight.float())
    half = rotary_dim // 2
    x1, x2 = xn[..., :half], xn[..., half:rotary_dim]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2, xn[..., rotary_dim:]), dim=-1)


def _to_fp8_storage(x32: torch.Tensor, kv_cache_dtype: str) -> torch.Tensor:
    if "e5m2" in kv_cache_dtype:
        return x32.clamp(-_FP8_E5M2_MAX, _FP8_E5M2_MAX).to(
            torch.float8_e5m2
        ).view(torch.uint8)
    return x32.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(
        torch.float8_e4m3fn
    ).view(torch.uint8)


_TRITON_MODE: bool | None = None


def resolve_triton_mode(probe: bool = True) -> bool:
    """Decide once (worker init / first call) whether the triton kernel runs
    the fused op. Self-verified against this torch fallback on first use."""
    global _TRITON_MODE
    if _TRITON_MODE is not None:
        return _TRITON_MODE
    if not probe:
        return False
    # The probe calls this module's torch fallback as its reference. Keep
    # that call from recursively invoking the probe before a mode is chosen.
    _TRITON_MODE = False
    try:
        from vllm_fl.ops.minimax_m3_triton import probe_triton_m3_fused

        _TRITON_MODE = probe_triton_m3_fused()
    except Exception:
        _TRITON_MODE = False
    return _TRITON_MODE


def _fused_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    if _TRITON_MODE is None:
        resolve_triton_mode()
    if _TRITON_MODE:
        from vllm_fl.ops.minimax_m3_triton import (
            triton_fused_minimax_m3_qknorm_rope_kv_insert as _triton_impl,
        )

        _triton_impl(
            qkv,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            num_heads,
            num_kv_heads,
            rotary_dim,
            eps,
            index_q_norm_weight,
            index_k_norm_weight,
            num_index_heads,
            slot_mapping,
            index_slot_mapping,
            kv_cache,
            index_cache,
            block_size,
            q_out,
            index_q_out,
            kv_cache_dtype,
        )
        return
    num_tokens = qkv.shape[0]
    nq, nkv, niq = int(num_heads), int(num_kv_heads), int(num_index_heads)
    has_index = niq > 0
    if num_tokens == 0:
        return

    qkv_v = qkv.view(num_tokens, -1, _HEAD_DIM)

    cs = cos_sin_cache.index_select(0, positions).float()
    half = rotary_dim // 2
    cos = cs[:, :half].unsqueeze(1)
    sin = cs[:, half:].unsqueeze(1)

    def norm_rope_slice(a: int, b: int, w: torch.Tensor) -> torch.Tensor:
        return _norm_rope(qkv_v[:, a:b].float(), w, eps, cos, sin, rotary_dim)

    # Q: gather into q_out when given, else rewrite in place.
    q_new = norm_rope_slice(0, nq, q_norm_weight)
    if q_out is not None:
        q_out.view(num_tokens, nq * _HEAD_DIM).copy_(
            q_new.reshape(num_tokens, -1).to(q_out.dtype)
        )
    else:
        qkv_v[:, :nq] = q_new.to(qkv.dtype)

    # K: always in place (also feeds the K-cache insert below).
    k_new = norm_rope_slice(nq, nq + nkv, k_norm_weight)
    k_new_dt = k_new.to(qkv.dtype)
    qkv_v[:, nq : nq + nkv] = k_new_dt

    # V: raw values, untouched in qkv.
    ik_new = None
    if has_index:
        iq_base = nq + 2 * nkv
        iq_new = norm_rope_slice(iq_base, iq_base + niq, index_q_norm_weight)
        if index_q_out is not None:
            if index_q_out.dtype == torch.float8_e4m3fn:
                index_q_out.view(num_tokens, niq * _HEAD_DIM).copy_(
                    iq_new.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
                    .to(torch.float8_e4m3fn)
                    .view(-1, niq * _HEAD_DIM)
                )
            else:
                index_q_out.view(num_tokens, niq * _HEAD_DIM).copy_(
                    iq_new.reshape(num_tokens, -1).to(index_q_out.dtype)
                )
        else:
            qkv_v[:, iq_base : iq_base + niq] = iq_new.to(qkv.dtype)

        ik_new = norm_rope_slice(
            iq_base + niq, iq_base + niq + 1, index_k_norm_weight
        )
        qkv_v[:, iq_base + niq : iq_base + niq + 1] = ik_new.to(qkv.dtype)

    if kv_cache is None:
        return

    # ── Paged cache inserts (sparse serving). ─────────────────────────────
    # During breakable-cudagraph capture this op runs eagerly on the capture
    # stream where host syncs are illegal, so boolean-mask selection
    # (nonzero() syncs) must be avoided: capture dummies use all-zero slots,
    # so write every row via clamped indices (harmless on the dummy buffers).
    # At replay the callable re-runs eagerly and the guarded path applies.
    capturing = torch.cuda.is_current_stream_capturing()
    mask = slot_mapping >= 0
    if capturing:
        sm_safe = slot_mapping.clamp(min=0)
        block = sm_safe // block_size
        token = sm_safe % block_size
        head_idx = torch.arange(nkv, device=qkv.device).view(1, nkv)
        if kv_cache.dtype == torch.uint8:
            k_store = _to_fp8_storage(k_new, kv_cache_dtype)
            v_store = _to_fp8_storage(
                qkv_v[:, nq + nkv : nq + 2 * nkv].float(), kv_cache_dtype
            )
        else:
            k_store = k_new_dt
            v_store = qkv_v[:, nq + nkv : nq + 2 * nkv]
        kv_cache[block.unsqueeze(1), 0, token.unsqueeze(1), head_idx, :] = k_store
        kv_cache[block.unsqueeze(1), 1, token.unsqueeze(1), head_idx, :] = v_store
    elif bool(mask.any()):
        sm = slot_mapping[mask]
        block = sm // block_size
        token = sm % block_size
        head_idx = torch.arange(nkv, device=qkv.device).view(1, nkv)

        if kv_cache.dtype == torch.uint8:
            k_store = _to_fp8_storage(k_new[mask], kv_cache_dtype)
            v_store = _to_fp8_storage(
                qkv_v[:, nq + nkv : nq + 2 * nkv].float()[mask], kv_cache_dtype
            )
        else:
            k_store = k_new_dt[mask]
            v_store = qkv_v[:, nq + nkv : nq + 2 * nkv][mask]

        kv_cache[block.unsqueeze(1), 0, token.unsqueeze(1), head_idx, :] = k_store
        kv_cache[block.unsqueeze(1), 1, token.unsqueeze(1), head_idx, :] = v_store

    idx_sm = (
        index_slot_mapping if index_slot_mapping is not None else slot_mapping
    )
    idx_mask = idx_sm >= 0
    head_dim = ik_new.shape[-1]
    flat_cache = index_cache.view(-1, head_dim)
    if index_cache.dtype == torch.float8_e4m3fn:
        vals = (
            ik_new.squeeze(1)
            .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
    else:
        vals = ik_new.squeeze(1).to(index_cache.dtype)
    if capturing:
        sel = idx_sm.clamp(min=0)
        if index_cache.dtype == torch.float8_e4m3fn:
            flat_cache.view(torch.uint8).index_copy_(
                0, sel, vals.view(torch.uint8)
            )
        else:
            flat_cache.index_copy_(0, sel, vals)
    elif bool(idx_mask.any()):
        sel_vals = vals[idx_mask]
        if index_cache.dtype == torch.float8_e4m3fn:
            flat_cache.view(torch.uint8).index_copy_(
                0, idx_sm[idx_mask], sel_vals.view(torch.uint8)
            )
        else:
            flat_cache.index_copy_(0, idx_sm[idx_mask], sel_vals)


def register_minimax_m3_ops() -> None:
    """Define the op schema + CUDA fallback when the native ext is absent."""
    if getattr(register_minimax_m3_ops, "_done", False):
        return

    try:
        import vllm._C  # noqa: F401
    except (ImportError, OSError):
        pass
    if hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert"):
        register_minimax_m3_ops._done = True
        return

    # The standalone native extension (vllm_fl._m3_fused_C) registers the
    # same op at import; the pure-torch fallback steps aside for it.
    try:
        import vllm_fl._m3_fused_C  # noqa: F401

        register_minimax_m3_ops._done = True
        return
    except (ImportError, OSError):
        pass

    lib = torch.library.Library("_C", "FRAGMENT")
    lib.define(SCHEMA)
    lib.impl(
        "fused_minimax_m3_qknorm_rope_kv_insert",
        _fused_minimax_m3_qknorm_rope_kv_insert,
        "CUDA",
    )
    # Keep the Library alive: a garbage-collected FRAGMENT library
    # unregisters its schemas and kernels with it.
    register_minimax_m3_ops._lib = lib
    register_minimax_m3_ops._done = True
