# Copyright (c) 2026 BAAI. All rights reserved.
"""Runtime compatibility patches for MiniMax-M3 on FL.

Upstream ``ApplyRotaryEmb.forward_static`` assumes ``cos``/``sin`` cover half
of ``head_size`` (full rotation). The MiniMax-M3 vision tower uses partial
RoPE (rotary_dim=78 < head_dim=128), so ``cos``/``sin`` only carry coefficients
for the rotated prefix. NVIDIA dispatches to the flash-attn rotary kernel,
which rotates the first ``2 * cos.shape[-1]`` dims and passes the rest
through; OOT platforms fall back to ``forward_native`` where the mismatched
broadcast (e.g. ``[2, N, H, 64] * [N, 1, 39]``) raises.

``_forward_static_partial`` slices the rotated prefix and concatenates the
pass-through suffix, mirroring the flash-attn kernel semantics. Full-rotation
callers (``2 * cos.shape[-1] >= head_size``) keep the upstream behavior.
"""

import torch

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

_original_forward_static = ApplyRotaryEmb.forward_static


def _forward_static_partial(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
    enable_fp32_compute: bool = False,
) -> torch.Tensor:
    rot_dim = 2 * cos.shape[-1]
    if rot_dim >= x.shape[-1]:
        return _original_forward_static(
            x, cos, sin, is_neox_style, enable_fp32_compute
        )

    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # The original helper handles the fp32 upcast, neox/gpt-j interleaving and
    # the final cast back to the input dtype for the rotated prefix.
    out_rot = _original_forward_static(
        x_rot, cos, sin, is_neox_style, enable_fp32_compute
    )
    return torch.cat([out_rot, x_pass.to(out_rot.dtype)], dim=-1)


def apply_patches() -> None:
    """Patch ApplyRotaryEmb for partial RoPE (idempotent across processes)."""
    if getattr(ApplyRotaryEmb, "_fl_partial_rope_patched", False):
        return
    ApplyRotaryEmb.forward_static = staticmethod(_forward_static_partial)
    ApplyRotaryEmb._fl_partial_rope_patched = True
