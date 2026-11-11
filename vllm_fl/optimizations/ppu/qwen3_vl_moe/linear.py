"""Decode-size text linears for Qwen3-VL-30B-A3B on PPU (VLLM_FL_QWEN3VL_DENSE=tuned).

Per-rank TP2 shapes: qkv 2048->2560, o_proj 2048->2048, MoE router 2048->128.
For M<=128 the generic Gems mm tile is 1.3-2.8x slower than the Gems
linear_kernel with a per-M tile (graph-replay timing, cold weights). Larger
M (prefill), other shapes, the LM head and the vision tower keep the original
path, so aten.mm / aten.linear stay on Gems there and their coverage is
unchanged.
"""

import torch

from ..minimax_m3.linear import _gems_linear

_original_linear_apply = None

_K = 2048
_PROJ_N = (2048, 2560)
_ROUTER_N = 128
# M upper bound -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_PROJ_TILES = ((16, (16, 32, 128, 4, 4)), (32, (32, 32, 128, 4, 4)),
               (64, (64, 64, 128, 4, 4)), (128, (64, 64, 128, 4, 3)))
_ROUTER_TILE = (16, 32, 256, 4, 3)


def tile_for(m, n):
    if n == _ROUTER_N:
        return _ROUTER_TILE
    return next(tile for bound, tile in _PROJ_TILES if m <= bound)


def handles(m, n, k):
    return 0 < m <= 128 and k == _K and (n == _ROUTER_N or n in _PROJ_N)


@torch.library.custom_op("vllm_fl::qwen3vl_bf16_linear", mutates_args=())
def bf16_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    if not handles(m, n, k):
        # Dynamic-shape graphs reach here with every M; prefill keeps Gems mm.
        return torch.nn.functional.linear(x, weight, bias)
    return _gems_linear(x, weight, bias, tile_for(m, n))


@bf16_linear.register_fake
def _bf16_linear_fake(x, weight, bias):
    return x.new_empty((x.shape[0], weight.shape[0]))


def _apply(self, layer, x, bias=None):
    weight = layer.weight
    if (x.ndim == 2 and x.dtype == weight.dtype == torch.bfloat16 and x.stride(-1) == 1
            and weight.is_contiguous() and weight.shape[1] == _K
            and (weight.shape[0] == _ROUTER_N or weight.shape[0] in _PROJ_N)):
        return bf16_linear(x, weight, bias)
    return _original_linear_apply(self, layer, x, bias)


def register_tuned_linear():
    global _original_linear_apply
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if _original_linear_apply is None:
        _original_linear_apply = UnquantizedLinearMethod.apply
        UnquantizedLinearMethod.apply = _apply
