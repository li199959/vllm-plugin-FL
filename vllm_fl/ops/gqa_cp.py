# Copyright (c) 2026 BAAI. All rights reserved.

"""Experimental CUDA GQA-CP helpers for Qwen3-style attention.

This is intentionally a conservative prefill-side optimization. It token-shards
the per-token QKV projection across tensor-parallel ranks, gathers the projected
local-head outputs back on the token dimension, then lets native vLLM attention
and KV-cache code run unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention as _Qwen3NextAttention,
)

from vllm_fl.ops.dsa_cp import local_token_shard

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_WARNED_MESSAGES: set[str] = set()


def _warning_once(message: str, *args: Any) -> None:
    key = message % args if args else message
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    logger.warning(message, *args)


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


def cuda_gqa_cp_enabled(vllm_config: Any | None = None) -> bool:
    additional_config = _additional_config(vllm_config)
    if "enable_gqa_cp" in additional_config:
        return _as_bool(additional_config.get("enable_gqa_cp"))

    env_value = os.environ.get("FL_ENABLE_GQA_CP")
    if env_value is not None:
        return _as_bool(env_value)
    return _as_bool(os.environ.get("VLLM_FL_ENABLE_GQA_CP"))


def cuda_gqa_cp_mode(vllm_config: Any | None = None) -> str:
    additional_config = _additional_config(vllm_config)
    value = additional_config.get("gqa_cp_mode")
    if value is None:
        value = os.environ.get("FL_GQA_CP_MODE")
    if value is None:
        value = os.environ.get("VLLM_FL_GQA_CP_MODE", "qkv_proj")
    return str(value).strip().lower()


def cuda_gqa_cp_min_tokens(vllm_config: Any | None = None) -> int:
    additional_config = _additional_config(vllm_config)
    value = additional_config.get("gqa_cp_min_tokens")
    if value is None:
        value = os.environ.get("FL_GQA_CP_MIN_TOKENS")
    if value is None:
        value = os.environ.get("VLLM_FL_GQA_CP_MIN_TOKENS", "1024")
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        _warning_once("Ignoring invalid gqa_cp_min_tokens value: %r", value)
        return 1024


def cuda_gqa_cp_is_prefill_producer(vllm_config: Any | None) -> bool:
    """Return whether this instance is allowed to run prefill GQA-CP.

    If there is no KV-transfer config, allow the optimization for local
    validation. In PD serving, restrict the default path to producer/both roles
    so decode-only consumers do not accidentally enter the experimental branch.
    """

    if vllm_config is None:
        return True
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None or getattr(kv_transfer_config, "kv_connector", None) is None:
        return True
    return bool(getattr(kv_transfer_config, "is_kv_producer", False))


def cuda_gqa_cp_graph_capture_enabled(vllm_config: Any | None) -> bool:
    if vllm_config is None:
        return False
    if getattr(vllm_config, "enforce_eager", False):
        return False
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "enforce_eager", False):
        return False
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return False
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    return cudagraph_mode is not None and cudagraph_mode != CUDAGraphMode.NONE


class CudaGQACPQwen3NextAttention(_Qwen3NextAttention):
    """Qwen3NextAttention with prefill-side token-parallel QKV projection."""

    _logged_enabled = False
    _logged_active = False
    _logged_disabled = False
    _logged_cudagraph_skip = False
    _logged_consumer_skip = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.vllm_config = get_current_vllm_config_or_none()
        self.cuda_gqa_cp_enabled = cuda_gqa_cp_enabled(self.vllm_config)
        self.cuda_gqa_cp_mode = cuda_gqa_cp_mode(self.vllm_config)
        self.cuda_gqa_cp_min_tokens = cuda_gqa_cp_min_tokens(self.vllm_config)
        self.cuda_gqa_cp_tp_size = 1
        self.cuda_gqa_cp_tp_rank = 0
        if self.cuda_gqa_cp_enabled:
            self.cuda_gqa_cp_tp_size = get_tensor_model_parallel_world_size()
            self.cuda_gqa_cp_tp_rank = get_tensor_model_parallel_rank()

        self.cuda_gqa_cp_skip_reason: str | None = None
        self.cuda_gqa_cp_active = (
            self.cuda_gqa_cp_enabled
            and self.cuda_gqa_cp_mode == "qkv_proj"
            and self.cuda_gqa_cp_tp_size > 1
        )

        if self.cuda_gqa_cp_active and not cuda_gqa_cp_is_prefill_producer(
            self.vllm_config
        ):
            self.cuda_gqa_cp_active = False
            self.cuda_gqa_cp_skip_reason = "this KV-transfer instance is not a producer"
            if not CudaGQACPQwen3NextAttention._logged_consumer_skip:
                logger.info(
                    "CUDA GQA-CP is enabled but skipped on decode/consumer instance."
                )
                CudaGQACPQwen3NextAttention._logged_consumer_skip = True

        if self.cuda_gqa_cp_active and cuda_gqa_cp_graph_capture_enabled(
            self.vllm_config
        ):
            self.cuda_gqa_cp_active = False
            self.cuda_gqa_cp_skip_reason = "CUDA graph capture is enabled"
            if not CudaGQACPQwen3NextAttention._logged_cudagraph_skip:
                logger.warning(
                    "CUDA GQA-CP qkv_proj mode is enabled, but CUDA graph capture "
                    "is enabled. Use --enforce-eager on the P/prefill instance."
                )
                CudaGQACPQwen3NextAttention._logged_cudagraph_skip = True

        if self.cuda_gqa_cp_enabled:
            if not CudaGQACPQwen3NextAttention._logged_enabled:
                logger.info(
                    "CUDA GQA-CP experimental Qwen attention wrapper enabled in "
                    "%s mode with min_tokens=%s.",
                    self.cuda_gqa_cp_mode,
                    self.cuda_gqa_cp_min_tokens,
                )
                CudaGQACPQwen3NextAttention._logged_enabled = True
            if (
                self.cuda_gqa_cp_active
                and self.cuda_gqa_cp_tp_rank == 0
                and not CudaGQACPQwen3NextAttention._logged_active
            ):
                logger.warning(
                    "CUDA GQA-CP qkv_proj ACTIVE: tp_rank=%s/%s, "
                    "q_heads=%s, kv_heads=%s, min_tokens=%s.",
                    self.cuda_gqa_cp_tp_rank,
                    self.cuda_gqa_cp_tp_size,
                    self.num_heads,
                    self.num_kv_heads,
                    self.cuda_gqa_cp_min_tokens,
                )
                CudaGQACPQwen3NextAttention._logged_active = True
        elif not CudaGQACPQwen3NextAttention._logged_disabled:
            logger.debug("CUDA GQA-CP Qwen attention wrapper registered but disabled.")
            CudaGQACPQwen3NextAttention._logged_disabled = True

    def _should_use_cuda_gqa_cp(self, num_tokens: int) -> bool:
        if not self.cuda_gqa_cp_active:
            return False
        if num_tokens < self.cuda_gqa_cp_min_tokens:
            return False
        if is_forward_context_available():
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
                return False
        return True

    def _qkv_proj_token_parallel(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        local_hidden_states = local_token_shard(
            hidden_states,
            self.cuda_gqa_cp_tp_size,
            self.cuda_gqa_cp_tp_rank,
        )
        local_qkv, _ = self.qkv_proj(local_hidden_states)
        qkv = tensor_model_parallel_all_gather(local_qkv, dim=0)
        return qkv[:num_tokens]

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        if self._should_use_cuda_gqa_cp(hidden_states.shape[0]):
            qkv = self._qkv_proj_token_parallel(hidden_states)
        else:
            qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


def apply_qwen_gqa_cp_patch() -> None:
    """Install the Qwen GQA-CP attention wrapper into upstream model modules."""

    import vllm.model_executor.models.qwen3_next as qwen3_next

    if getattr(qwen3_next, "_fl_gqa_cp_patched", False):
        return

    qwen3_next.Qwen3NextAttention = CudaGQACPQwen3NextAttention
    qwen3_next._fl_gqa_cp_patched = True

    try:
        import vllm.model_executor.models.qwen3_5 as qwen3_5

        qwen3_5.Qwen3NextAttention = CudaGQACPQwen3NextAttention
        qwen3_5._fl_gqa_cp_patched = True
    except Exception:
        # qwen3_5 may not be importable with older upstream versions.
        logger.debug("Qwen3.5 module not patched for CUDA GQA-CP", exc_info=True)

    logger.info("Registered CUDA GQA-CP patch for Qwen3Next/Qwen3.5 attention.")
