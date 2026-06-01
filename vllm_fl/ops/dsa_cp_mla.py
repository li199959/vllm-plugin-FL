# Copyright (c) 2026 BAAI. All rights reserved.

"""CUDA DSA-CP experimental MLA wrapper."""

from __future__ import annotations

import logging
from types import MethodType

import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)

from vllm_fl.ops.dsa_cp import (
    cuda_dsa_cp_a_proj_modes,
    cuda_dsa_cp_enabled,
    cuda_dsa_cp_indexer_proj_modes,
    cuda_dsa_cp_inspect,
    cuda_dsa_cp_inspect_layers,
    cuda_dsa_cp_layer_sharding,
    cuda_dsa_cp_mode,
    is_sparse_mla_model,
    local_token_shard,
    pad_first_dim as _pad_first_dim,
)

logger = logging.getLogger(__name__)


def _cuda_dsa_cp_indexer_proj_forward(
    self,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    tp_size = self._cuda_dsa_cp_tp_size
    tp_rank = self._cuda_dsa_cp_tp_rank
    if num_tokens == 0 or tp_size <= 1:
        return self._cuda_dsa_cp_original_forward(
            hidden_states, qr, positions, rotary_emb
        )

    tokens_per_rank = (num_tokens + tp_size - 1) // tp_size
    local_start = tp_rank * tokens_per_rank
    local_end = min(local_start + tokens_per_rank, num_tokens)

    local_hidden_states = hidden_states[local_start:local_end]
    local_qr = qr[local_start:local_end]
    local_positions = positions[local_start:local_end]

    q, _ = self.wq_b(local_qr)
    q = q.view(-1, self.n_head, self.head_dim)
    q_pe, q_nope = torch.split(
        q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
    )

    kw, _ = self.wk_weights_proj(local_hidden_states)
    k = kw[:, : self.head_dim]
    weights = kw[:, self.head_dim :]

    k = self.k_norm(k)
    k_pe, k_nope = torch.split(
        k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
    )

    q_pe, k_pe = rotary_emb(local_positions, q_pe, k_pe.unsqueeze(1))
    q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
    k_pe = k_pe.reshape(-1, 1, self.rope_dim)

    q = torch.cat([q_pe, q_nope], dim=-1)
    k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

    q = _pad_first_dim(q, tokens_per_rank)
    k = _pad_first_dim(k, tokens_per_rank)
    weights = _pad_first_dim(weights, tokens_per_rank)

    q = tensor_model_parallel_all_gather(q, dim=0)[:num_tokens]
    k = tensor_model_parallel_all_gather(k, dim=0)[:num_tokens]
    weights = tensor_model_parallel_all_gather(weights, dim=0)[:num_tokens]

    q = q.view(-1, self.head_dim)
    q_fp8, q_scale = per_token_group_quant_fp8(
        q,
        self.quant_block_size,
        column_major_scales=False,
        use_ue8m0=self.scale_fmt is not None,
    )
    q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
    q_scale = q_scale.view(-1, self.n_head, 1)

    weights = (
        weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
    )
    weights = weights.squeeze(-1)

    return self.indexer_op(hidden_states, q_fp8, k, weights)


class CudaDSACPMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    """Drop-in MLA wrapper with CUDA DSA-CP feature gating."""

    _logged_enabled = False
    _logged_disabled = False
    _logged_layer_sharding = False
    _logged_a_proj_active = False
    _logged_indexer_proj_active = False
    _logged_indexer_proj_cudagraph_skip = False
    _inspected_prefixes: set[str] = set()

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
        self.cuda_dsa_cp_inspect = cuda_dsa_cp_inspect(self.vllm_config)
        self.cuda_dsa_cp_inspect_layers = cuda_dsa_cp_inspect_layers(self.vllm_config)
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
            and self.cuda_dsa_cp_mode in cuda_dsa_cp_a_proj_modes()
            and self.cuda_dsa_cp_sparse_model
            and self.cuda_dsa_cp_tp_size > 1
            and self.q_lora_rank is not None
            and self.fused_qkv_a_proj is not None
        )
        self.cuda_dsa_cp_indexer_proj_active = (
            self.cuda_dsa_cp_enabled
            and self.cuda_dsa_cp_mode in cuda_dsa_cp_indexer_proj_modes()
            and self.cuda_dsa_cp_sparse_model
            and self.cuda_dsa_cp_tp_size > 1
            and self.indexer is not None
        )
        self.cuda_dsa_cp_indexer_proj_skip_reason = None

        if (
            self.cuda_dsa_cp_indexer_proj_active
            and self._cuda_graph_capture_enabled(self.vllm_config)
        ):
            self.cuda_dsa_cp_indexer_proj_active = False
            self.cuda_dsa_cp_indexer_proj_skip_reason = (
                "CUDA graph capture is enabled"
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
                self.cuda_dsa_cp_a_proj_active
                and self.cuda_dsa_cp_tp_rank == 0
                and not CudaDSACPMultiHeadLatentAttentionWrapper._logged_a_proj_active
            ):
                logger.warning(
                    "CUDA DSA-CP a_proj ACTIVE: prefix=%s, tp_rank=%s/%s, "
                    "q_lora_rank=%s, kv_lora_rank=%s.",
                    prefix,
                    self.cuda_dsa_cp_tp_rank,
                    self.cuda_dsa_cp_tp_size,
                    self.q_lora_rank,
                    self.kv_lora_rank,
                )
                CudaDSACPMultiHeadLatentAttentionWrapper._logged_a_proj_active = True
            if self.cuda_dsa_cp_indexer_proj_active:
                self._patch_indexer_proj_token_parallel()
            if (
                self.cuda_dsa_cp_indexer_proj_active
                and self.cuda_dsa_cp_tp_rank == 0
                and not CudaDSACPMultiHeadLatentAttentionWrapper._logged_indexer_proj_active
            ):
                logger.warning(
                    "CUDA DSA-CP indexer_proj ACTIVE: prefix=%s, tp_rank=%s/%s. "
                    "The indexer projection GEMMs are token-sharded; native sparse "
                    "indexer metadata and FlashMLA execution are preserved.",
                    prefix,
                    self.cuda_dsa_cp_tp_rank,
                    self.cuda_dsa_cp_tp_size,
                )
                CudaDSACPMultiHeadLatentAttentionWrapper._logged_indexer_proj_active = True
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
            if (
                self.cuda_dsa_cp_mode in cuda_dsa_cp_a_proj_modes()
                and not self.cuda_dsa_cp_a_proj_active
            ):
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
            if (
                self.cuda_dsa_cp_mode in cuda_dsa_cp_indexer_proj_modes()
                and not self.cuda_dsa_cp_indexer_proj_active
            ):
                if self.cuda_dsa_cp_indexer_proj_skip_reason is not None:
                    if (
                        self.cuda_dsa_cp_tp_rank == 0
                        and not CudaDSACPMultiHeadLatentAttentionWrapper._logged_indexer_proj_cudagraph_skip
                    ):
                        logger.warning(
                            "CUDA DSA-CP indexer_proj mode is enabled, but %s. "
                            "Keeping CUDA graphs enabled and falling back to "
                            "vLLM native sparse indexer execution. Use "
                            "--enforce-eager only when validating indexer_proj.",
                            self.cuda_dsa_cp_indexer_proj_skip_reason,
                        )
                        CudaDSACPMultiHeadLatentAttentionWrapper._logged_indexer_proj_cudagraph_skip = True
                else:
                    logger.warning(
                        "CUDA DSA-CP indexer_proj mode is enabled for %s but cannot "
                        "activate (sparse=%s, tp_size=%s, indexer=%s). Falling back "
                        "to vLLM native sparse indexer execution.",
                        prefix,
                        self.cuda_dsa_cp_sparse_model,
                        self.cuda_dsa_cp_tp_size,
                        self.indexer is not None,
                    )
            if self.cuda_dsa_cp_inspect and self.cuda_dsa_cp_tp_rank == 0:
                self._log_phase2_inspection(prefix)
        elif not CudaDSACPMultiHeadLatentAttentionWrapper._logged_disabled:
            logger.debug("CUDA DSA-CP wrapper registered but disabled.")
            CudaDSACPMultiHeadLatentAttentionWrapper._logged_disabled = True

    @staticmethod
    def _layer_index_from_prefix(prefix: str) -> int | None:
        marker = ".layers."
        if marker not in prefix:
            return None
        suffix = prefix.split(marker, 1)[1]
        try:
            return int(suffix.split(".", 1)[0])
        except ValueError:
            return None

    @staticmethod
    def _cuda_graph_capture_enabled(vllm_config) -> bool:
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

    @staticmethod
    def _module_summary(name: str, module: torch.nn.Module | None) -> str:
        if module is None:
            return f"{name}=None"

        params = []
        for param_name, param in module.named_parameters(recurse=True):
            params.append(f"{param_name}{tuple(param.shape)}")
            if len(params) >= 4:
                break
        if not params:
            params.append("no_params")

        attrs = []
        for attr_name in (
            "input_size",
            "output_size",
            "output_partition_sizes",
            "gather_output",
            "skip_bias_add",
            "tp_size",
        ):
            if hasattr(module, attr_name):
                attrs.append(f"{attr_name}={getattr(module, attr_name)}")

        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            attrs.append(f"quant={quant_method.__class__.__name__}")

        attr_text = ", ".join(attrs) if attrs else "attrs=none"
        return (
            f"{name}={module.__class__.__name__}"
            f"(params=[{'; '.join(params)}], {attr_text})"
        )

    def _log_phase2_inspection(self, prefix: str) -> None:
        layer_idx = self._layer_index_from_prefix(prefix)
        if self.cuda_dsa_cp_inspect_layers >= 0:
            if layer_idx is not None and layer_idx >= self.cuda_dsa_cp_inspect_layers:
                return
            if (
                layer_idx is None
                and len(CudaDSACPMultiHeadLatentAttentionWrapper._inspected_prefixes)
                >= self.cuda_dsa_cp_inspect_layers
            ):
                return

        if prefix in CudaDSACPMultiHeadLatentAttentionWrapper._inspected_prefixes:
            return
        CudaDSACPMultiHeadLatentAttentionWrapper._inspected_prefixes.add(prefix)

        logger.warning(
            "CUDA DSA-CP phase2 inspect: prefix=%s, sparse=%s, tp=%s/%s, "
            "hidden_size=%s, num_heads=%s, q_lora_rank=%s, kv_lora_rank=%s, "
            "layer_sharding=%s.",
            prefix,
            self.cuda_dsa_cp_sparse_model,
            self.cuda_dsa_cp_tp_rank,
            self.cuda_dsa_cp_tp_size,
            self.hidden_size,
            self.num_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.cuda_dsa_cp_layer_sharding,
        )
        logger.warning(
            "CUDA DSA-CP phase2 modules: %s | %s | %s | %s",
            self._module_summary("fused_qkv_a_proj", self.fused_qkv_a_proj),
            self._module_summary("q_b_proj", self.q_b_proj),
            self._module_summary("kv_b_proj", self.kv_b_proj),
            self._module_summary("o_proj", self.o_proj),
        )

    def _fused_qkv_a_proj_token_parallel(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        local_hidden_states = local_token_shard(
            hidden_states, self.cuda_dsa_cp_tp_size, self.cuda_dsa_cp_tp_rank
        )

        assert self.fused_qkv_a_proj is not None
        local_qkv_lora = self.fused_qkv_a_proj(local_hidden_states)[0]

        qkv_lora = tensor_model_parallel_all_gather(local_qkv_lora, dim=0)
        return qkv_lora[:num_tokens]

    def _patch_indexer_proj_token_parallel(self) -> None:
        indexer = self.indexer
        if indexer is None or getattr(
            indexer, "_cuda_dsa_cp_indexer_proj_patched", False
        ):
            return

        indexer._cuda_dsa_cp_tp_size = self.cuda_dsa_cp_tp_size
        indexer._cuda_dsa_cp_tp_rank = self.cuda_dsa_cp_tp_rank
        indexer._cuda_dsa_cp_original_forward = indexer.forward
        indexer.forward = MethodType(_cuda_dsa_cp_indexer_proj_forward, indexer)
        indexer._cuda_dsa_cp_indexer_proj_patched = True

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
