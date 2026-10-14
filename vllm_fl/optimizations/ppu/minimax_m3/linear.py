# Copyright (c) 2026 BAAI. All rights reserved.
"""Decode-size linears for MiniMax-M3 (VLLM_FL_M3_DENSE=hybrid).

Routing follows cold-weight (L2-missing) sweeps on PPU-ZW810E, see README:
  BF16, M<=128: K>=4096 skinny shapes use the plugin split-K kernel; other
                shapes use the Gems linear kernel with a per-M tile.
  FP32 router, M<=512 (all CUDA-graph sizes): plugin split-K (3xTF32).
  Everything larger stays native; linear/mm/addmm must remain blacklisted
  from global Gems dispatch.
"""

import torch
import triton

from .splitk import splitk_linear, splitk_linear_impl

_original_linear_apply = None

_M_BUCKETS = (1, 8, 16, 32, 64, 128)
# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) per M bucket.
_GEMS_TILES = {
    "small_k": {1: (16, 64, 64, 4, 4), 8: (16, 64, 64, 4, 4), 16: (16, 64, 64, 4, 4),
                32: (32, 128, 64, 4, 4), 64: (64, 64, 64, 4, 3), 128: (64, 128, 64, 4, 2)},
    "vocab": {1: (16, 64, 128, 4, 2), 8: (16, 64, 128, 4, 2), 16: (16, 64, 128, 4, 2),
              32: (32, 64, 128, 4, 3), 64: (64, 64, 128, 4, 3), 128: (64, 64, 128, 4, 3)},
    "large_k": {1: (16, 32, 256, 4, 4), 8: (16, 32, 256, 4, 4), 16: (16, 32, 256, 4, 4),
                32: (16, 64, 256, 4, 4), 64: (32, 32, 128, 4, 4), 128: (64, 64, 128, 4, 4)},
}


def _bucket(m):
    return next(b for b in _M_BUCKETS if m <= b)


def _use_splitk_bf16(m, n, k):
    return k >= 4096 and n <= 1024 and (m <= 64 or n <= 512)


def _gems_linear(x, weight, bias, tile):
    from flag_gems.ops.linear import linear_kernel

    m, k = x.shape
    n = weight.shape[0]
    bm, bn, bk, warps, stages = tile
    kernel = linear_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x, weight, weight if bias is None else bias, out,
        m, n, k, x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
        n, 1, 0 if bias is None else bias.stride(0), bias is not None,
        bm, bn, bk, num_warps=warps, num_stages=stages,
    )
    return out


@torch.library.custom_op("vllm_fl::m3_bf16_linear", mutates_args=())
def bf16_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    if m > 128:
        # Large dense prefill matmuls measured 15-40% slower in Gems.
        return torch.nn.functional.linear(x, weight, bias)
    if m == 0:
        return x.new_empty((0, n))
    if _use_splitk_bf16(m, n, k):
        return splitk_linear_impl(x, weight, bias)
    family = "vocab" if n > 8192 else ("small_k" if k <= 768 else "large_k")
    return _gems_linear(x, weight, bias, _GEMS_TILES[family][_bucket(m)])


@bf16_linear.register_fake
def _bf16_linear_fake(x, weight, bias):
    return x.new_empty((x.shape[0], weight.shape[0]))


def _hybrid_linear_apply(self, layer, x, bias=None):
    weight = layer.weight
    if x.ndim == 2 and weight.is_contiguous() and x.stride(-1) == 1:
        if x.dtype == weight.dtype == torch.bfloat16:
            return bf16_linear(x, weight, bias)
        # MoE router (GateLinear, fp32 weight) for every CUDA-graph size
        # (<=512). Split-K is 0.6-0.9x native at M=8-64; the rare mixed
        # steps at 72-512 cost more, but routing them too keeps the native
        # fp32 GEMM/GEMV kernels out of decode entirely.
        if (x.dtype == weight.dtype == torch.float32 and x.shape[0] <= 512
                and x.shape[1] >= 1024):
            return splitk_linear(x, weight, bias)
    return _original_linear_apply(self, layer, x, bias)


def register_hybrid_linear():
    global _original_linear_apply
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    if _original_linear_apply is None:
        _original_linear_apply = UnquantizedLinearMethod.apply
        UnquantizedLinearMethod.apply = _hybrid_linear_apply
