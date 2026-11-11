# Copyright (c) 2026 BAAI. All rights reserved.
"""Shared FlagGems linear launcher with an explicit tile configuration."""

import torch
import triton


def gems_linear(x, weight, bias, tile):
    from flag_gems.ops.linear import linear_kernel

    m, k = x.shape
    n = weight.shape[0]
    bm, bn, bk, warps, stages = tile
    kernel = linear_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x,
        weight,
        weight if bias is None else bias,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        n,
        1,
        0 if bias is None else bias.stride(0),
        bias is not None,
        bm,
        bn,
        bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out
