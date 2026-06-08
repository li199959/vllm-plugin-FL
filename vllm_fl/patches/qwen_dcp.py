# Copyright (c) 2026 BAAI. All rights reserved.

"""Qwen DCP enablement helpers.

This patch does not implement a new attention kernel. It wires Qwen3-Next /
Qwen3.5-style models to the existing FL attention backend when DCP is enabled,
and optionally derives ``decode_context_parallel_size`` from environment
variables before vLLM validates the model/parallel config.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_PATCHED = False

_QWEN_DCP_MODEL_TYPES = {
    "qwen3_next",
    "qwen3_5_text",
    "qwen3_5_moe_text",
    "qwen3_6_text",
    "qwen3_6_moe_text",
}

_QWEN_DCP_ARCHITECTURES = {
    "Qwen3NextForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    value_str = str(value).strip().lower()
    if value_str in _TRUE_VALUES:
        return True
    if value_str in _FALSE_VALUES:
        return False
    return None


def _additional_config(vllm_config: Any | None) -> dict[str, Any]:
    if vllm_config is None:
        return {}
    config = getattr(vllm_config, "additional_config", None)
    return config if isinstance(config, dict) else {}


def _get_first_config_value(vllm_config: Any | None, keys: tuple[str, ...]) -> Any:
    additional_config = _additional_config(vllm_config)
    for key in keys:
        if key in additional_config:
            return additional_config[key]
    for key in keys:
        env_value = os.environ.get(f"FL_{key.upper()}")
        if env_value is not None:
            return env_value
        env_value = os.environ.get(f"VLLM_FL_{key.upper()}")
        if env_value is not None:
            return env_value
    return None


def _hf_text_config_from_model_config(model_config: Any | None) -> Any | None:
    if model_config is None:
        return None
    return getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )


def is_qwen_dcp_model(model_config: Any | None) -> bool:
    text_config = _hf_text_config_from_model_config(model_config)
    model_type = getattr(text_config, "model_type", None)
    if model_type in _QWEN_DCP_MODEL_TYPES:
        return True

    architectures = getattr(model_config, "architectures", None) or getattr(
        text_config, "architectures", ()
    )
    return any(arch in _QWEN_DCP_ARCHITECTURES for arch in architectures)


def qwen_dcp_enabled(vllm_config: Any | None = None) -> bool:
    value = _get_first_config_value(
        vllm_config,
        ("enable_qwen_dcp", "qwen_dcp"),
    )
    parsed = _as_bool(value)
    return bool(parsed)


def qwen_dcp_size_value(vllm_config: Any | None = None) -> Any:
    return _get_first_config_value(
        vllm_config,
        ("qwen_dcp_size", "qwen_decode_context_parallel_size"),
    )


def _parse_dcp_size(value: Any) -> int | str | None:
    if value is None or value == "":
        return None
    value_str = str(value).strip().lower()
    if value_str == "auto":
        return "auto"
    try:
        size = int(value_str)
    except ValueError as exc:
        raise ValueError(f"Invalid Qwen DCP size: {value!r}") from exc
    if size < 1:
        raise ValueError(f"Qwen DCP size must be >= 1, got {size}.")
    return size


def _get_qwen_attention_shape(model_config: Any | None) -> tuple[int, int]:
    text_config = _hf_text_config_from_model_config(model_config)
    total_q_heads = getattr(text_config, "num_attention_heads", None)
    total_kv_heads = getattr(text_config, "num_key_value_heads", None)
    if total_kv_heads is None:
        total_kv_heads = total_q_heads

    if total_q_heads is None or total_kv_heads is None:
        raise ValueError(
            "Qwen DCP requires num_attention_heads and num_key_value_heads "
            "in the model config."
        )
    return int(total_q_heads), int(total_kv_heads)


def _valid_dcp_sizes(
    tp_size: int,
    total_q_heads: int,
    total_kv_heads: int,
) -> list[int]:
    if tp_size <= total_kv_heads:
        return []

    max_dcp_size = tp_size // total_kv_heads
    num_q_per_kv = total_q_heads // total_kv_heads
    return [
        size
        for size in range(2, max_dcp_size + 1)
        if tp_size % size == 0 and num_q_per_kv % size == 0
    ]


def choose_qwen_dcp_size(vllm_config: Any, requested: int | str | None) -> int:
    parallel_config = vllm_config.parallel_config
    tp_size = int(parallel_config.tensor_parallel_size)
    total_q_heads, total_kv_heads = _get_qwen_attention_shape(
        vllm_config.model_config
    )
    valid_sizes = _valid_dcp_sizes(tp_size, total_q_heads, total_kv_heads)

    if requested is None:
        requested = int(getattr(parallel_config, "decode_context_parallel_size", 1))

    if requested == "auto":
        if not valid_sizes:
            raise ValueError(
                "Cannot auto-enable Qwen DCP: no valid DCP size found. "
                f"tp_size={tp_size}, total_q_heads={total_q_heads}, "
                f"total_kv_heads={total_kv_heads}. Non-MLA DCP requires "
                "tp_size > total_kv_heads and num_q_per_kv divisible by dcp_size."
            )
        return max(valid_sizes)

    dcp_size = int(requested)
    if dcp_size <= 1:
        return dcp_size

    if dcp_size not in valid_sizes:
        raise ValueError(
            "Invalid Qwen DCP configuration: "
            f"dcp_size={dcp_size}, tp_size={tp_size}, "
            f"total_q_heads={total_q_heads}, total_kv_heads={total_kv_heads}. "
            f"Valid dcp sizes: {valid_sizes or 'none'}. For non-MLA GQA DCP, "
            "tp_size must be greater than total_kv_heads and both "
            "tp_size and num_q_per_kv must be divisible by dcp_size."
        )
    return dcp_size


def _maybe_apply_env_tuning(vllm_config: Any) -> None:
    parallel_config = vllm_config.parallel_config

    comm_backend = _get_first_config_value(vllm_config, ("qwen_dcp_comm_backend",))
    if comm_backend:
        comm_backend = str(comm_backend).strip().lower()
        if comm_backend not in ("ag_rs", "a2a"):
            raise ValueError(
                "qwen_dcp_comm_backend must be 'ag_rs' or 'a2a', "
                f"got {comm_backend!r}."
            )
        parallel_config.dcp_comm_backend = comm_backend

    interleave = _get_first_config_value(vllm_config, ("qwen_dcp_interleave_size",))
    if interleave not in (None, ""):
        interleave_size = int(str(interleave).strip())
        if interleave_size < 1:
            raise ValueError(
                f"qwen_dcp_interleave_size must be >= 1, got {interleave_size}."
            )
        parallel_config.cp_kv_cache_interleave_size = interleave_size


def apply_qwen_dcp_config(vllm_config: Any) -> None:
    model_config = getattr(vllm_config, "model_config", None)
    if not is_qwen_dcp_model(model_config):
        return

    requested_size = _parse_dcp_size(qwen_dcp_size_value(vllm_config))
    existing_size = int(vllm_config.parallel_config.decode_context_parallel_size)
    enabled = qwen_dcp_enabled(vllm_config)

    if not enabled and requested_size is None and existing_size <= 1:
        return

    if enabled and requested_size is None and existing_size <= 1:
        requested_size = "auto"

    dcp_size = choose_qwen_dcp_size(vllm_config, requested_size)
    vllm_config.parallel_config.decode_context_parallel_size = dcp_size
    _maybe_apply_env_tuning(vllm_config)

    if dcp_size > 1:
        os.environ["VLLM_FL_FORCE_FL_ATTENTION"] = "1"
        logger.info_once(
            "Qwen DCP enabled: dcp_size=%s, dcp_comm_backend=%s, "
            "cp_kv_cache_interleave_size=%s. Using FL attention backend.",
            dcp_size,
            vllm_config.parallel_config.dcp_comm_backend,
            vllm_config.parallel_config.cp_kv_cache_interleave_size,
        )


def should_use_qwen_dcp_attention_backend() -> bool:
    if _as_bool(os.environ.get("VLLM_FL_FORCE_FL_ATTENTION")):
        return True

    try:
        from vllm.config import get_current_vllm_config_or_none
    except Exception:
        return False

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False
    if not is_qwen_dcp_model(getattr(vllm_config, "model_config", None)):
        return False
    return int(vllm_config.parallel_config.decode_context_parallel_size) > 1


def apply_platform_patches() -> None:
    """Install a VllmConfig pre-validation hook for Qwen DCP env handling."""

    global _PATCHED
    if _PATCHED:
        return

    from vllm.config import VllmConfig

    original_post_init = VllmConfig.__post_init__

    def patched_post_init(self):
        apply_qwen_dcp_config(self)
        return original_post_init(self)

    VllmConfig.__post_init__ = patched_post_init
    _PATCHED = True

