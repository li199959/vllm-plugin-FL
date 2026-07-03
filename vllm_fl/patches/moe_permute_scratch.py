# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

"""Backport vLLM PR #43014 MoE permute scratch reuse.

The CUDA kernels introduced by vLLM #43014 live in vLLM's ``_moe_C``
extension.  This module only enables the Python-side scratch reuse when those
runtime symbols are available, and otherwise leaves the original implementation
untouched.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PATCHED = False
_SCRATCH_CACHE: dict[tuple[Any, ...], "MoEPermuteScratch"] = {}


def _scratch_ops_available() -> bool:
    moe_ops = getattr(torch.ops, "_moe_C", None)
    if moe_ops is None:
        return False
    for op_name in (
        "moe_permute_with_scratch",
        "moe_permute_sort_workspace_size",
        "moe_permute_unpermute_supported",
    ):
        if not hasattr(moe_ops, op_name):
            return False
    try:
        return bool(moe_ops.moe_permute_unpermute_supported())
    except Exception:
        return False


@dataclass
class MoEPermuteScratch:
    max_num_tokens: int
    topk: int
    num_experts: int
    num_local_experts: int
    device: torch.device
    hidden_size: int | None = None
    hidden_dtype: torch.dtype | None = None
    token_expert_indices: torch.Tensor = field(init=False)
    expert_first_token_offset: torch.Tensor = field(init=False)
    permuted_idx: torch.Tensor = field(init=False)
    inv_permuted_idx: torch.Tensor = field(init=False)
    permuted_hidden_states: torch.Tensor | None = field(init=False, default=None)
    sort_workspace: torch.Tensor = field(init=False)
    permuted_experts_id: torch.Tensor = field(init=False)
    sorted_row_idx: torch.Tensor = field(init=False)
    topk_ids_int32: torch.Tensor = field(init=False)
    topk_ids_for_sort: torch.Tensor = field(init=False)
    max_expanded_rows: int = field(init=False)

    def __post_init__(self) -> None:
        assert self.max_num_tokens > 0
        assert self.topk > 0
        assert self.num_experts > 0
        assert self.num_local_experts > 0
        if self.hidden_size is None:
            assert self.hidden_dtype is None
        else:
            assert self.hidden_dtype is not None

        self.max_expanded_rows = self.max_num_tokens * self.topk
        self.token_expert_indices = torch.arange(
            self.max_expanded_rows, dtype=torch.int32, device=self.device
        )
        self.expert_first_token_offset = torch.empty(
            self.num_local_experts + 1, dtype=torch.int64, device=self.device
        )
        self.permuted_idx = torch.empty(
            self.max_expanded_rows, dtype=torch.int32, device=self.device
        )
        self.inv_permuted_idx = torch.empty(
            self.max_expanded_rows, dtype=torch.int32, device=self.device
        )
        if self.hidden_size is not None:
            hidden_numel = self.max_expanded_rows * self.hidden_size
            self.permuted_hidden_states = torch.empty(
                hidden_numel, dtype=self.hidden_dtype, device=self.device
            )
        self.permuted_experts_id = torch.empty(
            self.max_expanded_rows, dtype=torch.int32, device=self.device
        )
        self.sorted_row_idx = torch.empty(
            self.max_expanded_rows, dtype=torch.int32, device=self.device
        )
        self.topk_ids_int32 = torch.empty(
            self.max_expanded_rows, dtype=torch.int32, device=self.device
        )
        self.topk_ids_for_sort = torch.empty(
            self.max_expanded_rows, dtype=torch.int32, device=self.device
        )
        sorter_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            self.max_expanded_rows, self.num_experts
        )
        self.sort_workspace = torch.empty(
            sorter_size, dtype=torch.int8, device=self.device
        )
        self.device = self.token_expert_indices.device

    def validate(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor) -> None:
        n_token, n_hidden = hidden_states.shape
        assert hidden_states.device == self.device
        assert topk_ids.device == self.device
        assert n_token <= self.max_num_tokens
        assert topk_ids.size(1) == self.topk
        assert topk_ids.size(0) == n_token
        if self.hidden_size is not None:
            assert n_hidden == self.hidden_size
            assert hidden_states.dtype == self.hidden_dtype
            assert self.permuted_hidden_states is not None

    def token_expert_indices_view(self, n_token: int) -> torch.Tensor:
        return self.token_expert_indices[: n_token * self.topk].view(n_token, self.topk)

    def prepare_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if topk_ids.dtype == torch.int32:
            return topk_ids
        numel = topk_ids.numel()
        topk_ids_int32 = self.topk_ids_int32[:numel].view_as(topk_ids)
        topk_ids_int32.copy_(topk_ids)
        return topk_ids_int32


def _cache_key(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    n_expert: int,
    n_local_expert: int,
    cache_hidden_states: bool,
) -> tuple[Any, ...]:
    device = hidden_states.device
    return (
        device.type,
        device.index,
        topk_ids.size(1),
        n_expert,
        n_local_expert,
        hidden_states.size(1) if cache_hidden_states else None,
        hidden_states.dtype if cache_hidden_states else None,
    )


def _get_auto_scratch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    n_expert: int,
    n_local_expert: int,
    cache_hidden_states: bool,
) -> MoEPermuteScratch:
    key = _cache_key(
        hidden_states, topk_ids, n_expert, n_local_expert, cache_hidden_states
    )
    scratch = _SCRATCH_CACHE.get(key)
    if scratch is None or scratch.max_num_tokens < hidden_states.size(0):
        scratch = MoEPermuteScratch(
            max_num_tokens=hidden_states.size(0),
            topk=topk_ids.size(1),
            num_experts=n_expert,
            num_local_experts=n_local_expert,
            device=hidden_states.device,
            hidden_size=hidden_states.size(1) if cache_hidden_states else None,
            hidden_dtype=hidden_states.dtype if cache_hidden_states else None,
        )
        _SCRATCH_CACHE[key] = scratch
    return scratch


def _make_moe_permute(original_moe_permute):
    def moe_permute(
        hidden_states: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        topk_ids: torch.Tensor,
        n_expert: int,
        n_local_expert: int = -1,
        expert_map: torch.Tensor | None = None,
        permuted_hidden_states: torch.Tensor | None = None,
        scratch: MoEPermuteScratch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        if n_local_expert == -1:
            n_local_expert = n_expert

        if scratch is None and _scratch_ops_available():
            scratch = _get_auto_scratch(
                hidden_states,
                topk_ids,
                n_expert,
                n_local_expert,
                cache_hidden_states=permuted_hidden_states is None,
            )

        if scratch is None:
            return original_moe_permute(
                hidden_states,
                a1q_scale,
                topk_ids,
                n_expert,
                n_local_expert,
                expert_map,
                permuted_hidden_states,
            )

        n_token, n_hidden = hidden_states.size()
        topk = topk_ids.size(1)
        assert (n_hidden * hidden_states.element_size()) % 16 == 0, (
            "permue kernel need hidden dim align to 16B"
        )
        permuted_row_size = n_token * topk

        if permuted_hidden_states is None:
            scratch.validate(hidden_states, topk_ids)
            hidden_numel = permuted_row_size * n_hidden
            scratch_hidden_states = scratch.permuted_hidden_states
            assert scratch_hidden_states is not None
            permuted_hidden_states = scratch_hidden_states[:hidden_numel].view(
                permuted_row_size, n_hidden
            )

        assert permuted_hidden_states.size() == (permuted_row_size, n_hidden), (
            f"Expected permuted hidden states to be {(permuted_row_size, n_hidden)}"
            f" but got {permuted_hidden_states.size()}"
        )

        scratch.validate(hidden_states, topk_ids)
        assert n_expert == scratch.num_experts
        assert n_local_expert == scratch.num_local_experts
        token_expert_indices = scratch.token_expert_indices_view(n_token)
        expert_first_token_offset = scratch.expert_first_token_offset
        permuted_idx = scratch.permuted_idx[:permuted_row_size]
        permuted_idx.fill_(permuted_row_size)
        inv_permuted_idx = scratch.inv_permuted_idx[:permuted_row_size].view(
            n_token, topk
        )
        permuted_experts_id = scratch.permuted_experts_id[:permuted_row_size].view(
            n_token, topk
        )
        sorted_row_idx = scratch.sorted_row_idx[:permuted_row_size].view(n_token, topk)
        topk_ids_for_sort = scratch.topk_ids_for_sort[:permuted_row_size].view(
            n_token, topk
        )
        topk_ids_int32 = scratch.prepare_topk_ids(topk_ids)

        torch.ops._moe_C.moe_permute_with_scratch(
            hidden_states,
            topk_ids_int32,
            token_expert_indices,
            expert_map,
            n_expert,
            n_local_expert,
            topk,
            permuted_hidden_states,
            expert_first_token_offset,
            inv_permuted_idx,
            permuted_idx,
            scratch.sort_workspace,
            permuted_experts_id,
            sorted_row_idx,
            topk_ids_for_sort,
        )

        if a1q_scale is not None and a1q_scale.dim() > 1:
            a1q_scale = a1q_scale[permuted_idx.clamp(max=n_token * topk - 1) // topk]

        return (
            permuted_hidden_states,
            a1q_scale,
            expert_first_token_offset,
            inv_permuted_idx.flatten(),
            permuted_idx,
        )

    return moe_permute


def apply_moe_permute_scratch_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    if os.environ.get("VLLM_FL_MOE_PERMUTE_SCRATCH", "1").lower() in (
        "0",
        "false",
        "no",
    ):
        return

    try:
        import vllm.model_executor.layers.fused_moe.moe_permute_unpermute as mod
    except Exception as e:
        logger.debug("Failed to import MoE permute module for scratch patch: %s", e)
        return

    original_moe_permute = mod.moe_permute
    if "scratch" in inspect.signature(original_moe_permute).parameters:
        _PATCHED = True
        return

    patched_moe_permute = _make_moe_permute(original_moe_permute)
    mod.MoEPermuteScratch = MoEPermuteScratch
    mod.moe_permute = patched_moe_permute

    for module_name in (
        "vllm.model_executor.layers.fused_moe.experts.cutlass_moe",
        "vllm.model_executor.layers.fused_moe.experts.fused_humming_moe",
    ):
        expert_mod = sys.modules.get(module_name)
        if expert_mod is None:
            continue
        if getattr(expert_mod, "moe_permute", None) is original_moe_permute:
            expert_mod.moe_permute = patched_moe_permute
        if not hasattr(expert_mod, "MoEPermuteScratch"):
            expert_mod.MoEPermuteScratch = MoEPermuteScratch

    _PATCHED = True
    logger.info(
        "Applied vLLM #43014 compatible MoE permute scratch patch "
        "(active only when _moe_C scratch ops exist)."
    )
