# Copyright (c) 2026 BAAI. All rights reserved.
"""Small-batch Qwen3-VL-Embedding-8B linears on a tuned Gems linear_kernel.

LLM linears (bf16, no bias, TP1): qkv 4096->6144, o 4096->4096,
gate_up 4096->24576, down 12288->4096. On PPU the Gems linear_kernel matches
cuBLAS up to M=256 (0.95-1.01x; gate_up 1.13-1.17x) and is 12-26% slower from
M=1024 (tools/ppu/qwen3_vl_embedding/sweep_linear.py). The serving config
blacklists Gems mm/addmm/linear (cuBLAS); batches with at most
VLLM_FL_Q3VLEMB_GEMS_SMALL_M tokens go to Gems with the swept tile.
"""

import os

import torch

from ..minimax_m3.linear import _gems_linear

_original_linear_apply = None

# (N, K) of each projection.
SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096), "down": (4096, 12288)}
_BIG = 1 << 62
# M upper bound -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages): fastest tile from
# the sweep (bf16, no bias, M 64 / 256).
_TILES = {
    "qkv": ((64, (64, 64, 32, 4, 4)), (_BIG, (128, 64, 64, 4, 3))),
    "o": ((64, (64, 64, 64, 4, 4)), (_BIG, (128, 256, 32, 8, 4))),
    "gate_up": ((64, (64, 128, 64, 4, 4)), (_BIG, (128, 128, 32, 4, 4))),
    "down": ((64, (64, 128, 64, 4, 4)), (_BIG, (128, 256, 32, 8, 4))),
}
_BY_SHAPE = {shape: name for name, shape in SHAPES.items()}
_small_m = 0


def small_m_from_env() -> int:
    return int(os.environ.get("VLLM_FL_Q3VLEMB_GEMS_SMALL_M", "0") or 0)


def tile_for(name, m):
    return next(tile for bound, tile in _TILES[name] if m <= bound)


def projection(x, weight):
    """Projection name if this linear may go to Gems, else None."""
    if not (_small_m > 0 and x.ndim == 2 and x.dtype == weight.dtype == torch.bfloat16
            and x.stride(-1) == 1 and weight.is_contiguous()):
        return None
    return _BY_SHAPE.get(tuple(weight.shape))


@torch.library.custom_op("vllm_fl::q3vlemb_gems_linear", mutates_args=())
def gems_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    m = x.shape[0]
    if m == 0:
        return x.new_empty((0, weight.shape[0]))
    if m > _small_m:
        # M is symbolic when compiled, so the size split happens here at run time.
        return torch.nn.functional.linear(x, weight, bias)
    return _gems_linear(x, weight, bias, tile_for(_BY_SHAPE[tuple(weight.shape)], m))


@gems_linear.register_fake
def _gems_linear_fake(x, weight, bias):
    return x.new_empty((x.shape[0], weight.shape[0]))


def _apply(self, layer, x, bias=None):
    if projection(x, layer.weight) is not None:
        return gems_linear(x, layer.weight, bias)
    return _original_linear_apply(self, layer, x, bias)


def register_hybrid_linear():
    global _original_linear_apply, _small_m
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    _small_m = small_m_from_env()
    if _original_linear_apply is None:
        _original_linear_apply = UnquantizedLinearMethod.apply
        UnquantizedLinearMethod.apply = _apply
