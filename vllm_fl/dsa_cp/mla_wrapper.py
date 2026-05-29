import logging

import torch
import torch.nn.functional as F

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

from vllm_fl.dsa_cp import is_deepseek_v32

logger = logging.getLogger(__name__)


def _round_up(x: int, divisor: int) -> int:
    return (x + divisor - 1) // divisor * divisor


class DSACPMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    """DSA-CP (Disaggregated Serving Architecture - Context Parallel) MLA wrapper.

    Optimization: fused_qkv_a_proj is replicated across TP ranks (same weight,
    same output for same input). Instead of each rank redundantly computing it
    for ALL tokens, each rank computes only 1/TP of the tokens, then all-gather
    the result. This saves (tp_size-1)/tp_size of the largest matmul in MLA
    attention (hidden_size × (q_lora_rank + kv_lora_rank + rope_dim)).

    Everything after all-gather proceeds identically to the base class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tp_size = get_tensor_model_parallel_world_size()
        self._tp_rank = get_tensor_model_parallel_rank()
        self._dsa_cp_active = (
            self._tp_size > 1
            and self.q_lora_rank is not None
            and is_deepseek_v32()
        )
        self._logged_first_forward = False
        if self._tp_rank == 0:
            logger.warning(
                "DSA-CP MLA wrapper init: tp_size=%d, active=%s, prefix=%s",
                self._tp_size, self._dsa_cp_active, self.prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._dsa_cp_active:
            return super().forward(positions, hidden_states, llama_4_scaling)

        num_tokens = hidden_states.shape[0]

        if num_tokens <= self._tp_size:
            return super().forward(positions, hidden_states, llama_4_scaling)

        if not self._logged_first_forward:
            self._logged_first_forward = True
            if self._tp_rank == 0:
                logger.warning(
                    "DSA-CP forward activated: num_tokens=%d, tp_size=%d",
                    num_tokens, self._tp_size,
                )

        # === DSA-CP: split A-projection across TP ranks ===
        num_tokens_pad = _round_up(num_tokens, self._tp_size)
        tokens_per_rank = num_tokens_pad // self._tp_size
        local_start = self._tp_rank * tokens_per_rank
        local_end = min(local_start + tokens_per_rank, num_tokens)
        local_actual = local_end - local_start

        local_hidden = hidden_states[local_start:local_end]

        # Pad to tokens_per_rank if this rank has fewer actual tokens
        if local_actual < tokens_per_rank:
            pad_size = tokens_per_rank - local_actual
            local_hidden = F.pad(local_hidden, (0, 0, 0, pad_size))

        # A-projection on local tokens only (the big compute saving)
        local_qkv_lora = self.fused_qkv_a_proj(local_hidden)[0]

        # All-gather across TP ranks: [tokens_per_rank, 2112] per rank
        # → [num_tokens_pad, 2112]
        tp_group = get_tp_group()
        full_qkv_lora = tp_group.all_gather(local_qkv_lora, dim=0)

        # Trim padding
        full_qkv_lora = full_qkv_lora[:num_tokens]

        # === From here, proceed exactly as the base class ===
        q_c, kv_lora = full_qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]

        kv_c, k_pe = kv_lora.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim:], k_pe
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
