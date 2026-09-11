# Copyright (c) 2026 BAAI. All rights reserved.
"""Fused SwiGLU-OAI MoE activations — FlagOS triton kernels.

Single-kernel replacements for the pure-torch fallbacks that decompose into
a mul/sigmoid/clamp micro-kernel storm (~8 launches, ~0.8ms/step at 1k
decode on M3):

- ``SWIGLUOAI_UNINTERLEAVE`` (MiniMax-M3): w13 packed [all gates | all ups]
  → ``gate * sigmoid(α·gate) * (up + β)`` with gate clamped to +limit and
  up clamped to ±limit, computed in fp32 like SiluAndMulWithClamp's native
  path.
- ``SWIGLUUOAI`` (gpt-oss): gate/up interleaved → same math.

One program covers BLOCK tokens × INTER elements of the output; fp32 math;
single rounding back to the output dtype. Static shapes, no host branches:
cudagraph-capturable.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swigluoai_uninterleave_kernel(
    x_ptr,  # [M, 2*half] gate | up
    out_ptr,  # [M, half]
    M,
    half,
    row_stride,
    out_stride,
    alpha,
    beta,
    limit,
    INTER: tl.constexpr,  # elements per program along half
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rm = rows < M

    offs = tl.arange(0, INTER)
    m2 = rm[:, None] & (offs[None, :] < half)

    gate = tl.load(
        x_ptr + rows[:, None] * row_stride + offs[None, :], mask=m2, other=0.0
    ).to(tl.float32)
    up = tl.load(
        x_ptr + rows[:, None] * row_stride + half + offs[None, :],
        mask=m2,
        other=0.0,
    ).to(tl.float32)

    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    y = gate * tl.sigmoid(alpha * gate) * (up + beta)

    tl.store(
        out_ptr + rows[:, None] * out_stride + offs[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=m2,
    )


@triton.jit
def _swigluoai_interleaved_kernel(
    x_ptr,  # [M, 2*half] interleaved g0,u0,g1,u1,...
    out_ptr,  # [M, half]
    M,
    half,
    row_stride,
    out_stride,
    alpha,
    beta,
    limit,
    INTER: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rm = rows < M

    offs = tl.arange(0, INTER)
    m2 = rm[:, None] & (offs[None, :] < half)
    base = rows[:, None] * row_stride + offs[None, :] * 2

    gate = tl.load(base, mask=m2, other=0.0).to(tl.float32)
    up = tl.load(base + 1, mask=m2, other=0.0).to(tl.float32)

    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    y = gate * tl.sigmoid(alpha * gate) * (up + beta)

    tl.store(
        out_ptr + rows[:, None] * out_stride + offs[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=m2,
    )


def _launch(kernel, x, out, alpha, beta, limit):
    M, full = x.shape
    half = full // 2
    INTER = min(triton.next_power_of_2(max(half, 16)), 2048)
    BLOCK_M = max(1, min(triton.next_power_of_2(max(M, 1)), 8))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(half, INTER))
    kernel[grid](
        x,
        out,
        M,
        half,
        x.stride(0),
        out.stride(0),
        float(alpha),
        float(beta),
        float(limit),
        INTER=INTER,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return out


def swigluoai_uninterleave_flaggems(
    output: torch.Tensor,
    x: torch.Tensor,
    clamp_limit: float,
    alpha: float = 1.702,
    beta: float = 1.0,
) -> torch.Tensor:
    """M3 layout: [gates | ups] → gate·σ(α·gate)·(up+β), clamped, fp32."""
    _launch(_swigluoai_uninterleave_kernel, x, output, alpha, beta, clamp_limit)
    return output


def swigluoai_interleaved_flaggems(
    output: torch.Tensor,
    x: torch.Tensor,
    clamp_limit: float = 7.0,
    alpha: float = 1.702,
    beta: float = 1.0,
) -> torch.Tensor:
    """gpt-oss layout: [g0,u0,g1,u1,...] → same math."""
    _launch(_swigluoai_interleaved_kernel, x, output, alpha, beta, clamp_limit)
    return output
