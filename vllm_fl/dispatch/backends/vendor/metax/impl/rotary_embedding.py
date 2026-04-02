# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def rotary_embedding_maca(
    obj,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding using vLLM's CUDA implementation.
    """

    from vllm._custom_ops import rotary_embedding

    rotary_embedding(
        position_ids,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox_style,
    )
    return query, key
