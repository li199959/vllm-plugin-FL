# Copyright (c) 2025 BAAI. All rights reserved.

"""Compatibility patches for vLLM NIXL on hybrid GDN models.

vLLM 0.20.2's NIXL hybrid-state path assumes Mamba2 when decomposing
conv state into x/B/C RDMA reads. Qwen3.5/3.6 hybrid models expose
``mamba_type='gdn_attention'`` instead. For equal-TP P/D setups we can
still transfer the whole local conv state by splitting its contiguous DS
layout into three byte ranges, without relying on Mamba2 semantics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ContiguousConvSplitInfo:
    conv_size: int

    @property
    def x_bytes(self) -> int:
        return self.conv_size - 2 * self.b_bytes

    @property
    def b_bytes(self) -> int:
        return self.conv_size // 3

    @property
    def local_conv_offsets(self) -> list[tuple[int, int]]:
        xb = self.x_bytes
        bb = self.b_bytes
        return [(0, xb), (xb, bb), (xb + bb, self.conv_size - xb - bb)]

    def remote_conv_offsets(
        self, local_rank_offset: int, tp_ratio: int
    ) -> list[tuple[int, int]]:
        if local_rank_offset != 0 or tp_ratio != 1:
            raise NotImplementedError(
                "GDN NIXL conv-state fallback only supports equal TP between "
                "prefill and decode engines."
            )
        return self.local_conv_offsets


def apply_nixl_gdn_patch() -> None:
    try:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )
        import vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils as ssm_utils
    except Exception:
        logger.exception("Failed to import vLLM NIXL GDN patch dependencies")
        return

    original = ssm_utils.derive_mamba_conv_split
    if getattr(original, "_fl_gdn_patched", False):
        return

    def derive_mamba_conv_split_gdn_compat(mamba_spec, local_tp):
        if getattr(mamba_spec, "mamba_type", None) != "gdn_attention":
            return original(mamba_spec, local_tp)

        if not is_conv_state_dim_first():
            raise AssertionError(
                "GDN NIXL conv-state fallback requires "
                "VLLM_SSM_CONV_STATE_LAYOUT=DS"
            )

        conv_shape = mamba_spec.shapes[0]
        if len(conv_shape) != 2:
            raise AssertionError(f"Expected 2D conv state shape, got {conv_shape}")

        conv_dtype_size = torch.tensor([], dtype=mamba_spec.dtypes[0]).element_size()
        conv_size = int(conv_shape[0]) * int(conv_shape[1]) * conv_dtype_size
        log_warning = getattr(logger, "warning_once", logger.warning)
        log_warning(
            "Using experimental equal-TP NIXL fallback for GDN conv-state "
            "transfer: conv_size=%s bytes, local_tp=%s",
            conv_size,
            local_tp,
        )
        return _ContiguousConvSplitInfo(conv_size=conv_size)

    derive_mamba_conv_split_gdn_compat._fl_gdn_patched = True
    ssm_utils.derive_mamba_conv_split = derive_mamba_conv_split_gdn_compat

    try:
        import vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker as nixl_worker

        nixl_worker.derive_mamba_conv_split = derive_mamba_conv_split_gdn_compat
    except Exception:
        # If the worker module has not been imported yet, it will import the
        # patched function from ssm_conv_transfer_utils later.
        pass
