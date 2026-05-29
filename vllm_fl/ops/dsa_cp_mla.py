# Copyright (c) 2026 BAAI. All rights reserved.

"""CUDA DSA-CP experimental MLA wrapper.

The first iteration is deliberately correctness-first. It replaces vLLM's
``MultiHeadLatentAttentionWrapper`` only to add feature detection, config
plumbing, and a stable place for the upcoming token-parallel execution path.
Actual sparse MLA kernels still come from vLLM.
"""

from __future__ import annotations

import logging

import torch
from vllm.config import get_current_vllm_config_or_none
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
        self.cuda_dsa_cp_sparse_model = bool(mla_modules.is_sparse) or is_sparse_mla_model(self.vllm_config)

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
                    "This first version keeps vLLM FlashMLA sparse execution for correctness.",
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
        elif not CudaDSACPMultiHeadLatentAttentionWrapper._logged_disabled:
            logger.debug("CUDA DSA-CP wrapper registered but disabled.")
            CudaDSACPMultiHeadLatentAttentionWrapper._logged_disabled = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(positions, hidden_states, llama_4_scaling)
