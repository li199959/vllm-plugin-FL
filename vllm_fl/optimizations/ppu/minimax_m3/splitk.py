# Copyright (c) 2026 BAAI. All rights reserved.
"""Triton split-K skinny GEMM for M3 decode: y = x @ W^T (+ b).

Small-M GEMMs with a long K launch too few tiles to read W at full bandwidth.
Each program accumulates one K slice in fp32 into a workspace; a second
kernel sums the slices in a fixed order (deterministic, no atomics), adds the
bias and casts. FP32 inputs use 3xTF32 dots, which keep router logits at
fp32-level error.
This is a plugin kernel, not part of FlagGems.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _m3_splitk_gemm_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    k_per_split,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_begin = pid_s * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < k_end),
            other=0.0,
        )
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < k_end),
            other=0.0,
        )
        if IS_FP32:
            # 3xTF32 keeps fp32-level error (~1e-6 at K=6144, below the
            # native GEMM) and is 2-4x faster than "ieee" on PPU.
            acc += tl.dot(a, b, input_precision="tf32x3")
        else:
            acc += tl.dot(a, b)
    part = part_ptr + pid_s * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(part, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _m3_splitk_reduce_kernel(
    part_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    SPLIT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * M * N + offs, mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs % N, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def splitk_config(m, n, k, dtype):
    """(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT, num_warps) from cold-weight sweeps
    at K=6144 (fp32 N=128; bf16 N=384/1024)."""
    fp32 = dtype == torch.float32
    if not fp32 and n > 512:
        # QKV projection: [M, 6144] x [1024, 6144].
        if m <= 16:
            return 16, 64, 64, 8, 4, 3
        if m <= 32:
            return 32, 128, 64, 8, 4, 3
        return 64, 64, 128, 4, 4, 3
    if m <= 16:
        block_m = 16
    elif m <= 32:
        block_m = 32
    else:
        block_m = 32 if fp32 and m <= 64 else 64
    block_n = 32 if fp32 else 64
    block_k = 32 if fp32 and m > 32 else 64
    split = 16 if (fp32 or n <= 512) and m <= 64 else 8
    return block_m, block_n, block_k, split, 4


def splitk_linear_impl(x, weight, bias=None, config=None):
    m, k = x.shape
    n = weight.shape[0]
    assert x.stride(1) == 1 and weight.stride(1) == 1 and x.dtype == weight.dtype
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m == 0:
        return out
    config = config or splitk_config(m, n, k, x.dtype)
    block_m, block_n, block_k, split, warps = config[:5]
    stages = config[5] if len(config) > 5 else 3
    k_per_split = triton.cdiv(triton.cdiv(k, split), block_k) * block_k
    split = triton.cdiv(k, k_per_split)
    part = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    _m3_splitk_gemm_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n), split)](
        x,
        weight,
        part,
        m,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        k_per_split,
        block_m,
        block_n,
        block_k,
        x.dtype == torch.float32,
        num_warps=warps,
        num_stages=stages,
    )
    reduce_block = 1024
    _m3_splitk_reduce_kernel[(triton.cdiv(m * n, reduce_block),)](
        part,
        out if bias is None else bias,
        out,
        m,
        n,
        split,
        bias is not None,
        reduce_block,
        num_warps=4,
    )
    return out


@torch.library.custom_op("vllm_fl::m3_splitk_linear", mutates_args=())
def splitk_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    return splitk_linear_impl(x, weight, bias)


@splitk_linear.register_fake
def _splitk_linear_fake(x, weight, bias):
    return x.new_empty((x.shape[0], weight.shape[0]))
