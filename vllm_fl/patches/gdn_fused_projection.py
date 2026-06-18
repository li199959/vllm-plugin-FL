# SPDX-License-Identifier: Apache-2.0
"""Fuse Qwen3.5 GDN projection tensor preparation.

This adapts the contiguous-layout Triton fusion from SGLang PR #21019.
It replaces split/reshape/contiguous operations after the QKVZ and BA
projections with one kernel. GDN attention kernels are unchanged.
"""

from __future__ import annotations

import logging
import os

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

_PATCHED = False
_ELIGIBLE_INSTANCE_LOGGED = False


def _enabled() -> bool:
    return os.environ.get("VLLM_FL_ENABLE_GDN_FUSED_PROJECTION", "0").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@triton.jit
def _fused_qkvzba_split_reshape_contiguous_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    token_idx = tl.program_id(0)
    qk_head_idx = tl.program_id(1)

    v_heads_per_qk: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    total_q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    total_k: tl.constexpr = total_q
    total_v: tl.constexpr = NUM_HEADS_V * HEAD_V
    total_qkvz: tl.constexpr = total_q + total_k + total_v * 2
    total_ba: tl.constexpr = NUM_HEADS_V * 2
    total_qkv: tl.constexpr = total_q + total_k + total_v

    q_offsets = tl.arange(0, HEAD_QK)
    v_offsets = tl.arange(0, v_heads_per_qk * HEAD_V)

    q_src = mixed_qkvz + token_idx * total_qkvz + qk_head_idx * HEAD_QK
    k_src = q_src + total_q
    v_src = (
        mixed_qkvz
        + token_idx * total_qkvz
        + total_q
        + total_k
        + qk_head_idx * v_heads_per_qk * HEAD_V
    )
    z_src = v_src + total_v

    q_dst = mixed_qkv + token_idx * total_qkv + qk_head_idx * HEAD_QK
    k_dst = q_dst + total_q
    v_dst = (
        mixed_qkv
        + token_idx * total_qkv
        + total_q
        + total_k
        + qk_head_idx * v_heads_per_qk * HEAD_V
    )
    z_dst = z + token_idx * total_v + qk_head_idx * v_heads_per_qk * HEAD_V

    tl.store(q_dst + q_offsets, tl.load(q_src + q_offsets))
    tl.store(k_dst + q_offsets, tl.load(k_src + q_offsets))
    tl.store(v_dst + v_offsets, tl.load(v_src + v_offsets))
    tl.store(z_dst + v_offsets, tl.load(z_src + v_offsets))

    ba_head_base = qk_head_idx * v_heads_per_qk
    for head_offset in tl.static_range(v_heads_per_qk):
        local_head = ba_head_base + head_offset
        b_value = tl.load(mixed_ba + token_idx * total_ba + local_head)
        a_value = tl.load(mixed_ba + token_idx * total_ba + NUM_HEADS_V + local_head)
        tl.store(b + token_idx * NUM_HEADS_V + local_head, b_value)
        tl.store(a + token_idx * NUM_HEADS_V + local_head, a_value)


def fused_qkvzba_split_reshape_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    *,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare contiguous QKV, Z, B and A tensors in one Triton launch."""
    if mixed_qkvz.ndim != 2 or mixed_ba.ndim != 2:
        raise ValueError(
            "GDN fused projection expects 2D projection outputs, got "
            f"{mixed_qkvz.shape=} and {mixed_ba.shape=}."
        )
    if num_heads_v % num_heads_qk != 0:
        raise ValueError(
            f"num_heads_v={num_heads_v} must be divisible by "
            f"num_heads_qk={num_heads_qk}."
        )
    if mixed_qkvz.shape[0] != mixed_ba.shape[0]:
        raise ValueError(
            "GDN fused projection inputs must have the same token count, got "
            f"{mixed_qkvz.shape[0]} and {mixed_ba.shape[0]}."
        )
    if mixed_qkvz.device != mixed_ba.device:
        raise ValueError(
            "GDN fused projection inputs must be on the same device, got "
            f"{mixed_qkvz.device} and {mixed_ba.device}."
        )

    num_tokens = mixed_qkvz.shape[0]
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    expected_qkvz_dim = qkv_dim + num_heads_v * head_v
    expected_ba_dim = num_heads_v * 2
    if mixed_qkvz.shape[1] != expected_qkvz_dim:
        raise ValueError(
            "Unexpected GDN QKVZ projection width: "
            f"expected {expected_qkvz_dim}, got {mixed_qkvz.shape[1]}."
        )
    if mixed_ba.shape[1] != expected_ba_dim:
        raise ValueError(
            "Unexpected GDN BA projection width: "
            f"expected {expected_ba_dim}, got {mixed_ba.shape[1]}."
        )

    # Projection outputs are normally contiguous. Keep the helper safe when
    # called directly or by a quantization backend returning a strided view.
    mixed_qkvz = mixed_qkvz.contiguous()
    mixed_ba = mixed_ba.contiguous()

    mixed_qkv = torch.empty(
        (num_tokens, qkv_dim),
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        (num_tokens, num_heads_v, head_v),
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        (num_tokens, num_heads_v),
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)

    _fused_qkvzba_split_reshape_contiguous_kernel[(num_tokens, num_heads_qk)](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        NUM_HEADS_QK=num_heads_qk,
        NUM_HEADS_V=num_heads_v,
        HEAD_QK=head_qk,
        HEAD_V=head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


def apply_gdn_fused_projection_patch() -> None:
    """Apply the optional Qwen3.5 GDN projection fusion once."""
    global _PATCHED
    if _PATCHED or not _enabled():
        return

    from vllm.model_executor.layers.mamba.gdn_linear_attn import (
        GatedDeltaNetAttention,
    )
    from vllm.utils.torch_utils import _encode_layer_name

    original_init = GatedDeltaNetAttention.__init__
    original_forward_cuda = GatedDeltaNetAttention.forward_cuda
    if getattr(original_forward_cuda, "_fl_gdn_fused_projection_patched", False):
        _PATCHED = True
        return

    def patched_init(self, *args, **kwargs):
        global _ELIGIBLE_INSTANCE_LOGGED
        original_init(self, *args, **kwargs)
        eligible = (
            not self.gqa_interleaved_layout
            and not hasattr(self, "in_proj_qkv")
            and self.num_v_heads % self.num_k_heads == 0
        )
        self._fl_gdn_fused_projection_eligible = eligible
        if eligible and not _ELIGIBLE_INSTANCE_LOGGED:
            _ELIGIBLE_INSTANCE_LOGGED = True
            logger.warning(
                "FL GDN fused projection enabled for Qwen3.5: "
                "num_k_heads=%d, num_v_heads=%d, tp_size=%d, "
                "head_k_dim=%d, head_v_dim=%d",
                self.num_k_heads,
                self.num_v_heads,
                self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
            )

    def patched_forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        if not getattr(self, "_fl_gdn_fused_projection_eligible", False):
            return original_forward_cuda(self, hidden_states, output)

        num_tokens = hidden_states.size(0)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        mixed_ba, _ = self.in_proj_ba(hidden_states)

        local_num_k_heads = self.num_k_heads // self.tp_size
        local_num_v_heads = self.num_v_heads // self.tp_size
        mixed_qkv, z, b, a = fused_qkvzba_split_reshape_contiguous(
            mixed_qkvz,
            mixed_ba,
            num_heads_qk=local_num_k_heads,
            num_heads_v=local_num_v_heads,
            head_qk=self.head_k_dim,
            head_v=self.head_v_dim,
        )

        core_attn_out = torch.zeros(
            (num_tokens, local_num_v_heads, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        core_attn_out = core_attn_out.reshape(num_tokens, -1)
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    patched_forward_cuda._fl_gdn_fused_projection_patched = True
    patched_forward_cuda._fl_gdn_fused_projection_original = original_forward_cuda
    GatedDeltaNetAttention.__init__ = patched_init
    GatedDeltaNetAttention.forward_cuda = patched_forward_cuda

    _PATCHED = True
    logger.warning(
        "Applied optional GDN fused projection patch (SGLang #21019); "
        "set VLLM_FL_ENABLE_GDN_FUSED_PROJECTION=0 to disable"
    )
