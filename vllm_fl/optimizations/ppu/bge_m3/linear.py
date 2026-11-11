# Copyright (c) 2026 BAAI. All rights reserved.
"""Selected BGE-M3 FP16 linears with explicit FlagGems tiles.

Shapes: qkv 1024->3072, o 1024->1024, fc1 1024->4096, fc2 4096->1024.
VLLM_FL_BGEM3_GEMS_LINEARS selects projections; GEMS_SMALL_M can also
route every supported projection for small batches. Larger unselected
projections retain the native path (blacklist Gems mm/addmm/linear)."""

import os

import torch

from ..common.linear import gems_linear as _gems_linear

_original_linear_apply = None

# (N, K) of each projection.
SHAPES = {
    "qkv": (3072, 1024),
    "o": (1024, 1024),
    "fc1": (4096, 1024),
    "fc2": (1024, 4096),
}
_BIG = 1 << 62
# M upper bound -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages): fastest tile per
# projection from a sweep of 46 configs on PPU (fp16, bias, M 512..8192).
_TILES = {
    "qkv": (
        (512, (128, 64, 64, 4, 3)),
        (1024, (128, 256, 32, 8, 4)),
        (2048, (128, 128, 32, 4, 4)),
        (4096, (128, 256, 32, 8, 3)),
        (_BIG, (128, 128, 32, 4, 4)),
    ),
    "o": (
        (512, (64, 128, 64, 4, 3)),
        (1024, (128, 128, 32, 4, 3)),
        (_BIG, (128, 128, 32, 4, 4)),
    ),
    "fc1": (
        (512, (128, 128, 32, 4, 3)),
        (4096, (128, 128, 32, 4, 4)),
        (_BIG, (128, 256, 32, 8, 3)),
    ),
    "fc2": (
        (512, (128, 128, 32, 4, 4)),
        (1024, (128, 256, 32, 8, 4)),
        (_BIG, (128, 128, 32, 4, 4)),
    ),
}
_BY_SHAPE = {shape: name for name, shape in SHAPES.items()}
_selected: frozenset = frozenset()
# Batches with at most this many tokens put every projection on Gems. They are
# the CUDA-graph sizes and tail batches, a small share of GPU time, and otherwise
# add their own cuBLAS kernels (small tiles, split-K reduction).
_small_m = 0


def selected_from_env() -> frozenset:
    raw = os.environ.get("VLLM_FL_BGEM3_GEMS_LINEARS", "o")
    names = frozenset(p.strip().lower() for p in raw.split(",") if p.strip())
    unknown = names - SHAPES.keys()
    if unknown:
        raise ValueError(f"VLLM_FL_BGEM3_GEMS_LINEARS: unknown {sorted(unknown)}")
    return names


def small_m_from_env() -> int:
    return int(os.environ.get("VLLM_FL_BGEM3_GEMS_SMALL_M", "0") or 0)


def tile_for(name, m):
    return next(tile for bound, tile in _TILES[name] if m <= bound)


def projection(x, weight):
    """Projection name if this linear may go to Gems, else None."""
    if not (
        x.ndim == 2
        and x.dtype == weight.dtype == torch.float16
        and x.stride(-1) == 1
        and weight.is_contiguous()
    ):
        return None
    name = _BY_SHAPE.get(tuple(weight.shape))
    return name if name in _selected or (name and _small_m > 0) else None


@torch.library.custom_op("vllm_fl::bgem3_gems_linear", mutates_args=())
def gems_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    m = x.shape[0]
    if m == 0:
        return x.new_empty((0, weight.shape[0]))
    name = _BY_SHAPE[tuple(weight.shape)]
    if name not in _selected and m > _small_m:
        # M is symbolic when compiled, so the size split happens here at run time.
        return torch.nn.functional.linear(x, weight, bias)
    return _gems_linear(x, weight, bias, tile_for(name, m))


@gems_linear.register_fake
def _gems_linear_fake(x, weight, bias):
    return x.new_empty((x.shape[0], weight.shape[0]))


def _apply(self, layer, x, bias=None):
    if projection(x, layer.weight) is not None:
        return gems_linear(x, layer.weight, bias)
    return _original_linear_apply(self, layer, x, bias)


def register_hybrid_linear():
    global _original_linear_apply, _selected, _small_m
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    _selected = selected_from_env()
    _small_m = small_m_from_env()
    if _original_linear_apply is None:
        _original_linear_apply = UnquantizedLinearMethod.apply
        UnquantizedLinearMethod.apply = _apply
