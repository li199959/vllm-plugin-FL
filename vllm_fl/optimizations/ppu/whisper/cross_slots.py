# Copyright (c) 2026 BAAI. All rights reserved.
"""Batched cross-attention slot mapping for encoder-decoder models.

vLLM's ``_get_cross_slot_mapping`` loops over the requests that need their
cross-attention K/V written this step and issues six tiny device ops per request
(arange, //, %, gather, *, +) plus a cat. Every one goes through FlagGems'
Python dispatch (~150-750 us each on PPU), so steps that admit several new audios
spend milliseconds of CPU here before the model runs (the GPU idles). This
version computes the same int64 slot mapping with one gather over all active
requests: slot[r, i] = block_table[r, i // block_size] * block_size + i % block_size.
"""

import numpy as np
import torch

_installed = False


def _batched_cross_slot_mapping(encoder_seq_lens, block_table_tensor, kv_cache_spec, device):
    active = np.nonzero(encoder_seq_lens)[0]
    if active.size == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    block_size = kv_cache_spec.block_size
    lens = encoder_seq_lens[active]
    max_len = int(lens.max())
    num_blocks = -(-max_len // block_size)
    rows = torch.from_numpy(active.astype(np.int64)).to(device, non_blocking=True)
    blocks = block_table_tensor[rows, :num_blocks].to(torch.int64)  # [R, nb]
    pos = torch.arange(max_len, dtype=torch.int64, device=device)
    slots = blocks[:, pos // block_size] * block_size + pos % block_size  # [R, max_len]
    if (lens == max_len).all():
        return slots.reshape(-1)
    # Ragged lengths: keep each request's first lens[r] slots, in request order.
    keep = torch.from_numpy(np.arange(max_len)[None, :] < lens[:, None]).to(device, non_blocking=True)
    return slots[keep]


def install() -> None:
    global _installed
    if _installed:
        return
    from vllm.model_executor.layers.attention import cross_attention

    cross_attention._get_cross_slot_mapping = _batched_cross_slot_mapping
    _installed = True
