# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagGems Gemma RMSNorm for MiniMax-M3 (VLLM_FL_M3_GEMMA_NORM=flaggems)."""

import weakref

import torch
import triton


@torch.library.custom_op("vllm_fl::m3_gemma_rms_norm", mutates_args=())
def gemma_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    from flag_gems.ops.gemma_rms_norm import gemma_rms_norm_kernel

    assert x.ndim == 2 and weight.ndim == 1
    rows, width = x.shape
    assert width == weight.numel()
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    inv_rms = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if rows:
        gemma_rms_norm_kernel[(rows,)](
            out, inv_rms, x, weight,
            out.stride(0), out.stride(1), x.stride(0), x.stride(1),
            width, eps, triton.next_power_of_2(width), num_warps=4,
        )
    return out


@gemma_rms_norm.register_fake
def _gemma_rms_norm_fake(x, weight, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


# Gemma's (1 + weight) in fp32, computed once per weight tensor. Recomputing
# it per call cost two extra kernels per norm (~240 launches per decode step).
# Keyed by object identity; the weakref drops the entry when the weight dies,
# so a new tensor reusing the id or address never sees a stale value.
_shifted_weights: dict[int, tuple] = {}


def _version(weight):
    try:
        return weight._version
    except RuntimeError:  # inference tensors carry no version counter
        return -1


def _shifted_weight(weight: torch.Tensor) -> torch.Tensor:
    key = id(weight)
    entry = _shifted_weights.get(key)
    state = (weight.data_ptr(), _version(weight))
    if entry is not None and entry[0]() is weight and entry[1] == state:
        return entry[2]
    from flag_gems import add

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Gemma norm weight first seen during CUDA graph capture")
    # Keep Gemma's offset in fp32; adding it in bf16 changes trained weights.
    shifted = add(weight.to(torch.float32), 1.0)
    ref = weakref.ref(weight, lambda _, k=key: _shifted_weights.pop(k, None))
    _shifted_weights[key] = (ref, state, shifted)
    return shifted


@torch.library.custom_op(
    "vllm_fl::m3_gemma_fused_add_rms_norm", mutates_args=("x", "residual")
)
def gemma_fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    from flag_gems.fused.fused_add_rms_norm import fused_add_rms_norm_kernel

    assert x.ndim == 2 and x.shape == residual.shape
    rows, width = x.shape
    assert weight.ndim == 1 and width == weight.numel()
    if not rows:
        return
    shifted_weight = _shifted_weight(weight)
    # Use the one-pass Gems kernel even for width=6144. The generic loop
    # wrapper re-reads a rounded residual before normalizing large rows.
    fused_add_rms_norm_kernel[(rows,)](
        x, residual, shifted_weight,
        x.stride(0), x.stride(1), residual.stride(0), residual.stride(1),
        width, eps, triton.next_power_of_2(width), num_warps=4,
    )


@gemma_fused_add_rms_norm.register_fake
def _gemma_fused_add_rms_norm_fake(x, residual, weight, eps):
    return None


def _gemma_forward(self, x, residual=None):
    if residual is None:
        return gemma_rms_norm(x, self.weight, self.variance_epsilon)
    gemma_fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
    return x, residual


def register_gemma_norm() -> None:
    from vllm.models.minimax_m3.nvidia.model import MiniMAXGemmaRMSNorm

    MiniMAXGemmaRMSNorm.forward = _gemma_forward
