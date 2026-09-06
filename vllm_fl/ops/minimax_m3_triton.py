# Copyright (c) 2026 BAAI. All rights reserved.
"""Triton kernel for ``_C::fused_minimax_m3_qknorm_rope_kv_insert``.

Mirrors ``csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu``
(one program per (token, head-slot), like the CUDA warp-per-slot layout):

- Q  slots [0, nq)               : Gemma RMSNorm + partial NeoX RoPE
- K  slots [nq, nq+nkv)          : same + K-cache insert
- V  slots (insert mode only)    : raw value-cache insert
- IQ slots (sparse only)         : norm+RoPE, gathered into index_q_out
- IK slot  (sparse only)         : norm+RoPE + index-cache insert

Pure kernel launches with static shapes — CUDA-graph capturable, which lets
the op join the graphed region instead of the eager-break path used by the
pure-torch fallback. Invalid slots (< 0) are skipped in-kernel like CUDA.
"""

import torch
import triton
import triton.language as tl

_FP8_E4M3_MAX = 448.0


@triton.jit
def _f32_to_e4m3_bits(x):
    """Software fp32 -> e4m3 (satfinite, RNE) encoder.

    This stack's triton has no native fp8e4nv, so emit the byte pattern via
    integer ops and store through a uint8 view of the fp8 tensor.
    """
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    u_all = x.to(tl.int32, bitcast=True)
    sign = (u_all >> 31) << 7
    u = u_all & 0x7FFFFFFF  # magnitude bits
    e_f = ((u >> 23) & 255) - 127
    m_full = (u & 0x7FFFFF) | 0x800000  # implicit leading 1

    # normal: 3-bit mantissa with round-to-nearest-even on the dropped bits
    sh = 20  # 23 - 3
    mant = (m_full >> sh) & 7  # drop the implicit leading 1
    rem = m_full & ((1 << sh) - 1)
    half = 1 << (sh - 1)
    round_up = (rem > half) | ((rem == half) & ((mant & 1) == 1))
    mant = mant + tl.where(round_up, 1, 0)  # may carry into the exponent
    e4n = e_f + 7 + (mant >> 3)
    mant = mant & 7
    normal_bits = (e4n << 3) | mant
    is_normal = (e_f >= -6) & (e4n <= 15)

    # subnormal / zero: round(|x| * 2^9) into the 0..7 mantissa field.
    # A result of 8 carries into the smallest normal (0x08 = 2^-6).
    sub_bits = tl.minimum((tl.abs(x) * 512.0 + 0.5).to(tl.int32), 8)

    return sign | tl.where(is_normal, normal_bits, sub_bits)


@triton.jit
def _m3_qknorm_rope_kv_insert_kernel(
    qkv_ptr,  # [N, row_heads * 128] bf16/fp16
    q_out_ptr,  # [N, nq*128] or unused
    iq_out_ptr,  # [N, niq*128] (bf16 or e4m3) or unused
    q_norm_w_ptr,
    k_norm_w_ptr,
    iq_norm_w_ptr,
    ik_norm_w_ptr,
    cos_sin_ptr,  # [max_pos, rotary_dim]
    positions_ptr,  # [N] i64
    slot_ptr,  # [N] i64 or unused
    idx_slot_ptr,  # [N] i64 or unused
    kv_cache_ptr,  # [nb, 2, bs, nkv, 128] or unused
    index_cache_ptr,  # [nb*bs, 128] or unused
    kv_s0,
    kv_s1,
    kv_s2,
    kv_s3,
    nq,
    nkv,
    niq,
    rotary_dim,
    eps,
    block_size,
    HAS_INDEX: tl.constexpr,
    INSERT_KV: tl.constexpr,
    HAS_Q_OUT: tl.constexpr,
    HAS_IQ_OUT: tl.constexpr,
    FP8_IQ_OUT: tl.constexpr,
    FP8_INDEX_CACHE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)

    v_begin = nq + nkv
    iq_begin = nq + 2 * nkv
    ik_slot = iq_begin + niq  # only meaningful when HAS_INDEX

    # ── resolve role (uniform scalars per program) ─────────────────────────
    is_q = pid_s < nq
    is_k = (pid_s >= nq) and (pid_s < v_begin)
    is_v = INSERT_KV and (pid_s >= v_begin) and (pid_s < v_begin + nkv)
    is_iq = HAS_INDEX and (pid_s >= iq_begin) and (pid_s < ik_slot)
    is_ik = HAS_INDEX and (pid_s == ik_slot)

    # head index inside qkv (V packs right after K; IQ/IK after [q|k|v])
    if is_q or is_k:
        head = pid_s
    elif is_v:
        head = pid_s  # V section starts at (nq+nkv): slot offset == head
    elif is_iq:
        head = nq + 2 * nkv + (pid_s - iq_begin)
    else:  # ik
        head = nq + 2 * nkv + niq

    row_heads = nq + 2 * nkv + (niq + 1 if HAS_INDEX else 0)
    base = qkv_ptr + pid_t.to(tl.int64) * (row_heads * HEAD_DIM) + head * HEAD_DIM

    apply_norm = not is_v  # everything but V is normed
    do_rope = not is_v

    offs = tl.arange(0, HEAD_DIM)
    x = tl.load(base + offs).to(tl.float32)

    # ── Gemma RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + w), fp32 ────────────
    inv = 1.0
    if apply_norm:
        if is_q:
            w_ptr = q_norm_w_ptr
        elif is_k:
            w_ptr = k_norm_w_ptr
        elif is_iq:
            w_ptr = iq_norm_w_ptr
        else:
            w_ptr = ik_norm_w_ptr
        w = tl.load(w_ptr + offs).to(tl.float32)
        var = tl.sum(x * x, axis=0) / HEAD_DIM
        inv = 1.0 / tl.sqrt(var + eps)
        x = x * inv * (1.0 + w)

    # ── partial NeoX RoPE on [0, rotary_dim) ───────────────────────────────
    if do_rope:
        half = rotary_dim // 2
        # partner element (neox pairing i <-> i+half), pre-norm
        partner = tl.where(
            offs < half,
            offs + half,
            tl.where(offs < rotary_dim, offs - half, offs),
        )
        xp = tl.load(base + partner).to(tl.float32)
        if apply_norm:
            if is_q:
                wp_ptr = q_norm_w_ptr
            elif is_k:
                wp_ptr = k_norm_w_ptr
            elif is_iq:
                wp_ptr = iq_norm_w_ptr
            else:
                wp_ptr = ik_norm_w_ptr
            wp = tl.load(wp_ptr + partner).to(tl.float32)
            xp = xp * inv * (1.0 + wp)
        pos = tl.load(positions_ptr + pid_t)
        i_cos = tl.where(offs < half, offs, tl.where(offs < rotary_dim, offs - half, 0))
        cos = tl.load(
            cos_sin_ptr + pos * rotary_dim + i_cos,
            mask=offs < rotary_dim,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_ptr + pos * rotary_dim + half + i_cos,
            mask=offs < rotary_dim,
            other=0.0,
        ).to(tl.float32)
        x = tl.where(
            offs < half,
            x * cos - xp * sin,
            tl.where(offs < rotary_dim, x * cos + xp * sin, x),
        )

    out_ty = qkv_ptr.dtype.element_ty

    # ── stores ──────────────────────────────────────────────────────────────
    if is_q:
        if HAS_Q_OUT:
            tl.store(
                q_out_ptr + pid_t.to(tl.int64) * (nq * HEAD_DIM) + pid_s * HEAD_DIM + offs,
                x.to(q_out_ptr.dtype.element_ty),
            )
        else:
            tl.store(base + offs, x.to(out_ty))
    elif is_k:
        tl.store(base + offs, x.to(out_ty))
        if INSERT_KV:
            sm = tl.load(slot_ptr + pid_t)
            if sm >= 0:
                b = sm // block_size
                t = sm % block_size
                dst = (
                    kv_cache_ptr
                    + b * kv_s0
                    + 0 * kv_s1
                    + t * kv_s2
                    + (pid_s - nq) * kv_s3
                )
                tl.store(dst + offs, x.to(kv_cache_ptr.dtype.element_ty))
    elif is_v:
        if INSERT_KV:
            sm = tl.load(slot_ptr + pid_t)
            if sm >= 0:
                b = sm // block_size
                t = sm % block_size
                dst = (
                    kv_cache_ptr
                    + b * kv_s0
                    + 1 * kv_s1
                    + t * kv_s2
                    + (pid_s - v_begin) * kv_s3
                )
                tl.store(dst + offs, x.to(kv_cache_ptr.dtype.element_ty))
    elif is_iq:
        if HAS_IQ_OUT:
            dst = (
                iq_out_ptr
                + pid_t.to(tl.int64) * (niq * HEAD_DIM)
                + (pid_s - iq_begin) * HEAD_DIM
                + offs
            )
            if FP8_IQ_OUT:
                tl.store(dst, _f32_to_e4m3_bits(x).to(tl.uint8))
            else:
                tl.store(dst, x.to(iq_out_ptr.dtype.element_ty))
        else:
            tl.store(base + offs, x.to(out_ty))
    else:  # ik
        tl.store(base + offs, x.to(out_ty))
        if INSERT_KV:
            ism = tl.load(idx_slot_ptr + pid_t)
            if ism >= 0:
                dst = index_cache_ptr + ism.to(tl.int64) * HEAD_DIM
                if FP8_INDEX_CACHE:
                    tl.store(dst + offs, _f32_to_e4m3_bits(x).to(tl.uint8))
                else:
                    tl.store(dst + offs, x.to(index_cache_ptr.dtype.element_ty))


def triton_fused_minimax_m3_qknorm_rope_kv_insert(
    qkv,
    q_norm_weight,
    k_norm_weight,
    cos_sin_cache,
    positions,
    num_heads,
    num_kv_heads,
    rotary_dim,
    eps,
    index_q_norm_weight=None,
    index_k_norm_weight=None,
    num_index_heads=0,
    slot_mapping=None,
    index_slot_mapping=None,
    kv_cache=None,
    index_cache=None,
    block_size=0,
    q_out=None,
    index_q_out=None,
    kv_cache_dtype="auto",
) -> None:
    num_tokens = qkv.shape[0]
    nq, nkv, niq = int(num_heads), int(num_kv_heads), int(num_index_heads)
    has_index = niq > 0
    insert_kv = kv_cache is not None
    if num_tokens == 0:
        return
    if insert_kv:
        assert kv_cache.stride(4) == 1, "kv_cache head_dim must be contiguous"
        assert index_cache.stride(-1) == 1, "index_cache head_dim contiguous"
    assert rotary_dim % 2 == 0 and rotary_dim <= 128

    v_slots = nkv if insert_kv else 0
    grid = (num_tokens, nq + nkv + v_slots + (niq + 1 if has_index else 0))

    dummy = qkv  # never dereferenced when the matching constexpr is off
    # fp8 tensors are addressed through their uint8 bit-view (no native fp8
    # type in this stack's triton; the kernel software-encodes e4m3 bytes).
    iq_out_arg = (
        index_q_out.view(torch.uint8)
        if index_q_out is not None
        and index_q_out.dtype == torch.float8_e4m3fn
        else (index_q_out if index_q_out is not None else dummy)
    )
    index_cache_arg = (
        index_cache.view(torch.uint8)
        if index_cache is not None
        and index_cache.dtype == torch.float8_e4m3fn
        else (index_cache if (insert_kv and has_index) else dummy)
    )
    _m3_qknorm_rope_kv_insert_kernel[grid](
        qkv,
        q_out if q_out is not None else dummy,
        iq_out_arg,
        q_norm_weight,
        k_norm_weight,
        index_q_norm_weight if has_index else q_norm_weight,
        index_k_norm_weight if has_index else k_norm_weight,
        cos_sin_cache,
        positions,
        slot_mapping if slot_mapping is not None else positions,
        index_slot_mapping
        if index_slot_mapping is not None
        else (slot_mapping if slot_mapping is not None else positions),
        kv_cache if kv_cache is not None else qkv,
        index_cache_arg,
        kv_cache.stride(0) if insert_kv else 0,
        kv_cache.stride(1) if insert_kv else 0,
        kv_cache.stride(2) if insert_kv else 0,
        kv_cache.stride(3) if insert_kv else 0,
        nq,
        nkv,
        niq,
        int(rotary_dim),
        float(eps),
        int(block_size),
        HAS_INDEX=has_index,
        INSERT_KV=insert_kv,
        HAS_Q_OUT=q_out is not None,
        HAS_IQ_OUT=index_q_out is not None,
        FP8_IQ_OUT=index_q_out is not None
        and index_q_out.dtype == torch.float8_e4m3fn,
        FP8_INDEX_CACHE=index_cache is not None
        and index_cache.dtype == torch.float8_e4m3fn,
        HEAD_DIM=128,
        num_warps=1,
    )


def probe_triton_m3_fused() -> bool:
    """Runtime self-check: tiny end-to-end call verified against the pure-torch
    fallback. Called once at worker init (eager) to decide the impl and
    whether the eager-break wrap is needed."""
    import os

    if os.environ.get("VLLM_FL_M3_FUSED_IMPL", "triton") == "torch":
        return False
    try:
        from vllm_fl.ops.minimax_m3_ops import (
            _fused_minimax_m3_qknorm_rope_kv_insert as torch_impl,
        )

        torch.manual_seed(0)
        dev = torch.cuda.current_device()
        N, nq, nkv, niq, rot, eps = 5, 4, 2, 3, 64, 1e-6
        HD = 128
        row = (nq + 2 * nkv + niq + 1) * HD
        qkv0 = torch.randn(N, row, dtype=torch.bfloat16, device=dev)
        w = torch.randn(HD, dtype=torch.bfloat16, device=dev) * 0.1
        pos = torch.randint(0, 64, (N,), device=dev)
        inv = 1.0 / (
            5000000.0
            ** (torch.arange(0, rot, 2, dtype=torch.float32, device=dev) / rot)
        )
        t = torch.arange(64, dtype=torch.float32, device=dev)
        cs = torch.cat([torch.outer(t, inv).cos(), torch.outer(t, inv).sin()], -1).to(
            torch.bfloat16
        )
        bs, nb = 4, 16
        sm = torch.tensor([3, 9, -1, 17, 40], device=dev)
        ism = torch.tensor([4, 10, 12, 16, -1], device=dev)
        for fp8_idx in (False, True):
            kv_t = torch.zeros(nb, 2, bs, nkv, HD, dtype=torch.bfloat16, device=dev)
            kv_k = kv_t.clone()
            idx_dtype = torch.float8_e4m3fn if fp8_idx else torch.bfloat16
            idx_t = torch.zeros(nb, bs, HD, dtype=idx_dtype, device=dev)
            idx_k = idx_t.clone()
            q_out_t = torch.empty(N, nq * HD, dtype=torch.bfloat16, device=dev)
            q_out_k = q_out_t.clone()
            iq_t = torch.empty(N, niq * HD, dtype=idx_dtype, device=dev)
            iq_k = iq_t.clone()
            torch_impl(
                qkv0.clone(), w, w, cs, pos, nq, nkv, rot, eps, w, w, niq,
                sm, ism, kv_t, idx_t, bs, q_out_t, iq_t, "auto",
            )
            triton_fused_minimax_m3_qknorm_rope_kv_insert(
                qkv0.clone(), w, w, cs, pos, nq, nkv, rot, eps, w, w, niq,
                sm, ism, kv_k, idx_k, bs, q_out_k, iq_k, "auto",
            )
            torch.cuda.synchronize()
            for a, b in (
                (q_out_t, q_out_k),
                (iq_t, iq_k),
                (idx_t, idx_k),
                (kv_t, kv_k),
            ):
                if fp8_idx:
                    ok = torch.allclose(a.float(), b.float(), atol=0.5)
                else:
                    ok = torch.allclose(a.float(), b.float(), atol=2e-2)
                if not ok:
                    return False
        return True
    except Exception:
        return False
