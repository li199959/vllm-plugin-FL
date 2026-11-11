"""Decode text linears for Qwen3-VL-30B-A3B with explicit FlagGems tiles.

Supported BF16 TP2 projections: qkv 2048->2560, o 2048->2048,
router 2048->128, for batches of at most 128 tokens.
Other shapes and batch sizes retain their original implementation."""

import torch

from ..common.linear import gems_linear as _gems_linear

_original_linear_apply = None

_K = 2048
_PROJ_N = (2048, 2560)
_ROUTER_N = 128
# M upper bound -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_PROJ_TILES = (
    (16, (16, 32, 128, 4, 4)),
    (32, (32, 32, 128, 4, 4)),
    (64, (64, 64, 128, 4, 4)),
    (128, (64, 64, 128, 4, 3)),
)
_ROUTER_TILE = (16, 32, 256, 4, 3)


def tile_for(m, n):
    if n == _ROUTER_N:
        return _ROUTER_TILE
    return next(tile for bound, tile in _PROJ_TILES if m <= bound)


def handles(m, n, k):
    return 0 < m <= 128 and k == _K and (n == _ROUTER_N or n in _PROJ_N)


@torch.library.custom_op("vllm_fl::qwen3vl_bf16_linear", mutates_args=())
def bf16_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
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
    if (
        x.ndim == 2
        and x.dtype == weight.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and weight.is_contiguous()
        and weight.shape[1] == _K
        and (weight.shape[0] == _ROUTER_N or weight.shape[0] in _PROJ_N)
    ):
        return bf16_linear(x, weight, bias)
    return _original_linear_apply(self, layer, x, bias)


def register_tuned_linear():
    global _original_linear_apply
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if _original_linear_apply is None:
        _original_linear_apply = UnquantizedLinearMethod.apply
        UnquantizedLinearMethod.apply = _apply
