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
    if not getattr(ApplyRotaryEmb, "_fl_partial_rope_patched", False):
        ApplyRotaryEmb.forward_static = staticmethod(_forward_static_partial)
        ApplyRotaryEmb._fl_partial_rope_patched = True
    _apply_fused_op_eager_break()


def _apply_fused_op_eager_break() -> None:
    """Keep the M3 fused op out of breakable-cudagraph captures when the
    pure-torch fallback is in charge.

    With the triton kernel active (self-verified at worker init) the op
    normally joins the graphed region; set VLLM_FL_M3_FUSED_EAGER_BREAK=1
    to also break around the triton path (bisect/diagnostic for the
    capture-interaction crash).
    """
    import os

    from vllm_fl.ops.minimax_m3_ops import resolve_triton_mode

    if resolve_triton_mode() and os.environ.get(
        "VLLM_FL_M3_FUSED_EAGER_BREAK", "0"
    ) != "1":
        return  # triton kernel inside graphs (default)

    try:
        from vllm.compilation.breakable_cudagraph import (
            eager_break_during_capture,
        )
    except ImportError:
        return

    import vllm._custom_ops as ops

    fn = ops.fused_minimax_m3_qknorm_rope_kv_insert
    if getattr(fn, "_fl_eager_break", False):
        return
    wrapped = eager_break_during_capture(fn)
    wrapped._fl_eager_break = True
    ops.fused_minimax_m3_qknorm_rope_kv_insert = wrapped
