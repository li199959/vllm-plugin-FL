# Copyright (c) 2026 BAAI. All rights reserved.

"""CUDA DSA-CP experimental MLA wrapper."""

from __future__ import annotations

import logging

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper

from vllm_fl.ops.dsa_cp import (
    cuda_dsa_cp_enabled,
    cuda_dsa_cp_layer_sharding,
    cuda_dsa_cp_mode,
    is_sparse_mla_model,
)

logger = logging.getLogger(__name__)


class CudaDSACPMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    """Drop-in MLA wrapper with CUDA DSA-CP feature gating."""

    _logged_enabled = False
    _logged_disabled = False
    _logged_layer_sharding = False

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.vllm_config = get_current_vllm_config_or_none()
        self.cuda_dsa_cp_enabled = cuda_dsa_cp_enabled(self.vllm_config)
        self.cuda_dsa_cp_mode = cuda_dsa_cp_mode(self.vllm_config)
        self.cuda_dsa_cp_layer_sharding = cuda_dsa_cp_layer_sharding(self.vllm_config)
        self.cuda_dsa_cp_sparse_model = bool(
            mla_modules.is_sparse
        ) or is_sparse_mla_model(self.vllm_config)
        self.cuda_dsa_cp_tp_size = 1
        self.cuda_dsa_cp_tp_rank = 0
        if self.cuda_dsa_cp_enabled:
            self.cuda_dsa_cp_tp_size = get_tensor_model_parallel_world_size()
            self.cuda_dsa_cp_tp_rank = get_tensor_model_parallel_rank()
        self.cuda_dsa_cp_a_proj_active = (
            self.cuda_dsa_cp_enabled
            and self.cuda_dsa_cp_mode == "a_proj"
            and self.cuda_dsa_cp_sparse_model
            and self.cuda_dsa_cp_tp_size > 1
            and self.q_lora_rank is not None
            and self.fused_qkv_a_proj is not None
        )

        if self.cuda_dsa_cp_enabled and not self.cuda_dsa_cp_sparse_model:
            logger.warning(
                "CUDA DSA-CP was enabled for %s, but this MLA layer is not sparse. "
                "Falling back to vLLM native MLA execution.",
                prefix,
            )
            self.cuda_dsa_cp_enabled = False

        if self.cuda_dsa_cp_enabled:
            if not CudaDSACPMultiHeadLatentAttentionWrapper._logged_enabled:
                logger.info(
                    "CUDA DSA-CP experimental wrapper enabled in %s mode. "
                    "a_proj mode token-shards fused_qkv_a_proj and keeps vLLM FlashMLA sparse execution.",
                    self.cuda_dsa_cp_mode,
                )
                CudaDSACPMultiHeadLatentAttentionWrapper._logged_enabled = True
            if (
                self.cuda_dsa_cp_layer_sharding
                and not CudaDSACPMultiHeadLatentAttentionWrapper._logged_layer_sharding
            ):
                logger.warning(
                    "CUDA DSA-CP received layer_sharding=%s. Weight layer sharding is "
                    "not active in this first version; sparse MLA will run normally.",
                    self.cuda_dsa_cp_layer_sharding,
                )
                CudaDSACPMultiHeadLatentAttentionWrapper._logged_layer_sharding = True
            if self.cuda_dsa_cp_mode == "a_proj" and not self.cuda_dsa_cp_a_proj_active:
                logger.warning(
                    "CUDA DSA-CP a_proj mode is enabled for %s but cannot activate "
                    "(sparse=%s, tp_size=%s, q_lora_rank=%s, fused_qkv_a_proj=%s). "
                    "Falling back to vLLM native MLA execution.",
                    prefix,
                    self.cuda_dsa_cp_sparse_model,
                    self.cuda_dsa_cp_tp_size,
                    self.q_lora_rank,
                    self.fused_qkv_a_proj is not None,
                )
        elif not CudaDSACPMultiHeadLatentAttentionWrapper._logged_disabled:
            logger.debug("CUDA DSA-CP wrapper registered but disabled.")
            CudaDSACPMultiHeadLatentAttentionWrapper._logged_disabled = True

    def _fused_qkv_a_proj_token_parallel(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        tokens_per_rank = (
            num_tokens + self.cuda_dsa_cp_tp_size - 1
        ) // self.cuda_dsa_cp_tp_size
        local_start = self.cuda_dsa_cp_tp_rank * tokens_per_rank
        local_hidden_states = hidden_states[local_start : local_start + tokens_per_rank]

        assert self.fused_qkv_a_proj is not None
        local_qkv_lora = self.fused_qkv_a_proj(local_hidden_states)[0]
        local_num_tokens = local_qkv_lora.shape[0]
        if local_num_tokens < tokens_per_rank:
            padding = local_qkv_lora.new_empty(
                (
                    tokens_per_rank - local_num_tokens,
                    local_qkv_lora.shape[-1],
                )
            )
            local_qkv_lora = torch.cat((local_qkv_lora, padding), dim=0)

        qkv_lora = tensor_model_parallel_all_gather(local_qkv_lora, dim=0)
        return qkv_lora[:num_tokens]

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.cuda_dsa_cp_a_proj_active:
            return super().forward(positions, hidden_states, llama_4_scaling)

        assert self.q_lora_rank is not None
        assert self.q_a_layernorm is not None, (
            "q_a_layernorm is required when q_lora_rank is not None"
        )
        assert self.q_b_proj is not None, (
            "q_b_proj is required when q_lora_rank is not None"
        )

        qkv_lora = self._fused_qkv_a_proj_token_parallel(hidden_states)
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]
