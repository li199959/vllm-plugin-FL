# Copyright (c) 2026 BAAI. All rights reserved.
"""Fused sigmoid MoE routing (topk_sigmoid) — FlagOS triton kernel.

Mirrors upstream ``csrc/libtorch_stable/moe/topk_softmax_kernels.cu``
(SCORING_SIGMOID): scores = sigmoid(gating) with NaN clamped to 0; the
bias (e_score_correction_bias) participates in expert *selection* only;
weights are the unbiased scores; renormalize divides by the selected sum
(denominator clamped to 1 when not positive); ties break to the lowest
expert index like the CUDA kernel's index tie-break (an all-NaN row with
no bias selects experts [0, 1, ..., k-1]).

One program per token: the full expert row lives in registers (BLOCK =
next_pow2(num_experts)), so the k-loop is pure in-register argmax+mask —
a single kernel launch replacing the ~15 small kernels of the python
fallback. Static shapes, no host branches: cudagraph-capturable.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_sigmoid_kernel(
    gating_ptr,  # [M, E] (row-strided)
    bias_ptr,  # [E] or unused
    weights_ptr,  # [M, K] fp32 out
    indices_ptr,  # [M, K] int32/int64 out
    row_stride,
    num_experts,
    HAS_BIAS: tl.constexpr,
    RENORM: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < num_experts

    g = tl.load(
        gating_ptr + token * row_stride + offs, mask=m, other=0.0
    ).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    s = tl.where(s == s, s, 0.0)  # NaN scores clamp to 0 (CUDA parity)

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs, mask=m, other=0.0).to(tl.float32)
        choice = s + b
    else:
        choice = s
    choice = tl.where(m, choice, float("-inf"))

    sel_sum = 0.0
    for k in tl.static_range(K):
        # tl.argmax breaks ties toward the lowest index by default, matching
        # the CUDA kernel's index tie-break.
        idx = tl.argmax(choice, axis=0)
        w = tl.sum(tl.where(offs == idx, s, 0.0), axis=0)  # unbiased score
        if RENORM:
            sel_sum += w
        tl.store(weights_ptr + token * K + k, w)
        tl.store(indices_ptr + token * K + k, idx.to(indices_ptr.dtype.element_ty))
        choice = tl.where(offs == idx, float("-inf"), choice)

    if RENORM:
        denom = tl.where(sel_sum > 0.0, sel_sum, 1.0)
        for k in tl.static_range(K):
            ptr = weights_ptr + token * K + k
            tl.store(ptr, tl.load(ptr) / denom)


def topk_sigmoid_flaggems(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,  # scratch, unused (CUDA parity)
    gating_output: torch.Tensor,
    renormalize: bool = False,
    bias: torch.Tensor | None = None,
):
    """Out-of-place fused sigmoid routing; returns (weights, indices)."""
    M, E = gating_output.shape
    K = topk_weights.shape[-1]
    BLOCK = max(triton.next_power_of_2(E), 16)
    # num_warps=4: 1-warp wide-vector kernels miscompile under cudagraph
    # replay on this stack (see the M3 fused-op lesson).
    _topk_sigmoid_kernel[(M,)](
        gating_output,
        bias if bias is not None else gating_output,
        topk_weights,
        topk_indices,
        gating_output.stride(0),
        E,
        HAS_BIAS=bias is not None,
        RENORM=renormalize,
        K=K,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return topk_weights, topk_indices
