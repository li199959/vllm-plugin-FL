# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional
import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm_fl.dispatch import call_op


class RotaryEmbeddingFL(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base,
            is_neox_style, dtype
        )

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        positions = positions.flatten()
        num_tokens = positions.shape[0]

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
        key_shape = None
        key_rot = None
        key_pass = None
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., : self.rotary_dim]
            if self.rotary_dim < self.head_size:
                key_pass = key[..., self.rotary_dim:]

        q_embed, k_embed = call_op(
            "rotary_embedding",
            self,
            query_rot,
            key_rot,
            self.head_size,
            self.cos_sin_cache,
            positions,
            self.is_neox_style,
        )

        if self.rotary_dim < self.head_size:
            query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
            if k_embed is not None and key_pass is not None and key_shape is not None:
                key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
            else:
                key = None
        else:
            query = q_embed.reshape(query_shape)
            key = k_embed.reshape(key_shape) if k_embed is not None and key_shape is not None else None

        return query, key


__all__ = ["RotaryEmbeddingFL"]
