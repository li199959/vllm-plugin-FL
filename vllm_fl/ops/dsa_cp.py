# Copyright (c) 2026 BAAI. All rights reserved.

"""Experimental CUDA DSA-CP helpers.

This module intentionally keeps the first CUDA implementation conservative:
it wires the feature flag, validates that we are on a sparse MLA model, and
exposes token-shard planning without changing cache semantics yet. The actual
attention computation continues to use vLLM's FlashMLA sparse backend.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_WARNED_MESSAGES: set[str] = set()


def _warning_once(message: str, *args: Any) -> None:
    key = message % args if args else message
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    logger.warning(message, *args)


@dataclass(frozen=True)
class DSACPTokenPlan:
    """Token interval assigned to one tensor-parallel rank."""

    num_tokens: int
    padded_num_tokens: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    rank: int
    world_size: int

    @property
    def local_num_tokens(self) -> int:
        return max(0, self.local_end - self.local_start)

    @property
    def local_num_tokens_with_pad(self) -> int:
        return self.local_end_with_pad - self.local_start


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUE_VALUES


def _additional_config(vllm_config: Any | None) -> dict[str, Any]:
    if vllm_config is None:
        return {}
    config = getattr(vllm_config, "additional_config", None)
    return config if isinstance(config, dict) else {}


def cuda_dsa_cp_enabled(vllm_config: Any | None = None) -> bool:
    """Return whether the experimental CUDA DSA-CP wrapper is enabled."""

    additional_config = _additional_config(vllm_config)
    if "enable_dsa_cp" in additional_config:
        return _as_bool(additional_config.get("enable_dsa_cp"))
    if "enable_flashcomm1" in additional_config:
        return _as_bool(additional_config.get("enable_flashcomm1"))

    env_value = os.environ.get("VLLM_FL_ENABLE_DSA_CP")
    if env_value is not None:
        return _as_bool(env_value)

    # Compatibility with vLLM-Ascend deployment snippets. On CUDA this only
    # enables the FL experimental wrapper; it does not use Ascend kernels.
    return _as_bool(os.environ.get("VLLM_ASCEND_ENABLE_FLASHCOMM1"))


def cuda_dsa_cp_mode(vllm_config: Any | None = None) -> str:
    """Return the CUDA DSA-CP mode.

    ``safe`` is the initial mode: enable validation and telemetry while using
    vLLM's native sparse MLA execution for correctness.
    """

    additional_config = _additional_config(vllm_config)
    value = additional_config.get("dsa_cp_mode")
    if value is None:
        value = os.environ.get("VLLM_FL_DSA_CP_MODE", "safe")
    return str(value).strip().lower()


def cuda_dsa_cp_layer_sharding(vllm_config: Any | None = None) -> list[str]:
    additional_config = _additional_config(vllm_config)
    value = additional_config.get("layer_sharding", [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    _warning_once("Ignoring invalid layer_sharding config for CUDA DSA-CP: %r", value)
    return []


def is_sparse_mla_model(vllm_config: Any | None) -> bool:
    if vllm_config is None:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return bool(
        hasattr(hf_text_config, "index_topk")
        or hasattr(hf_config, "index_topk")
    )


def build_token_plan(num_tokens: int, world_size: int, rank: int) -> DSACPTokenPlan:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")

    padded_num_tokens = ((num_tokens + world_size - 1) // world_size) * world_size
    local_num_tokens = padded_num_tokens // world_size
    local_start = rank * local_num_tokens
    local_end_with_pad = local_start + local_num_tokens
    local_end = min(local_end_with_pad, num_tokens)
    return DSACPTokenPlan(
        num_tokens=num_tokens,
        padded_num_tokens=padded_num_tokens,
        local_start=local_start,
        local_end=local_end,
        local_end_with_pad=local_end_with_pad,
        rank=rank,
        world_size=world_size,
    )


def slice_first_dim_with_plan(tensor: torch.Tensor, plan: DSACPTokenPlan) -> torch.Tensor:
    return tensor[plan.local_start:plan.local_end]
