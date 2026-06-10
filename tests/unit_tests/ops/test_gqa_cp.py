# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.ops.dsa_cp import gather_first_dim_token_shards, local_token_shard
from vllm_fl.ops.gqa_cp import (
    cuda_gqa_cp_enabled,
    cuda_gqa_cp_is_prefill_producer,
    cuda_gqa_cp_min_tokens,
    cuda_gqa_cp_mode,
)


def _config(additional_config=None, kv_transfer_config=None):
    return SimpleNamespace(
        additional_config=additional_config,
        kv_transfer_config=kv_transfer_config,
    )


def test_cuda_gqa_cp_enabled_from_additional_config():
    cfg = _config(additional_config={"enable_gqa_cp": True})

    assert cuda_gqa_cp_enabled(cfg)


def test_cuda_gqa_cp_mode_defaults_to_qkv_proj():
    assert cuda_gqa_cp_mode(_config()) == "qkv_proj"


def test_cuda_gqa_cp_min_tokens_from_additional_config():
    cfg = _config(additional_config={"gqa_cp_min_tokens": "4096"})

    assert cuda_gqa_cp_min_tokens(cfg) == 4096


@pytest.mark.parametrize(
    ("kv_transfer_config", "expected"),
    [
        (None, True),
        (SimpleNamespace(kv_connector=None, is_kv_producer=False), True),
        (SimpleNamespace(kv_connector="FlagCXConnector", is_kv_producer=True), True),
        (SimpleNamespace(kv_connector="FlagCXConnector", is_kv_producer=False), False),
    ],
)
def test_cuda_gqa_cp_prefill_producer_role(kv_transfer_config, expected):
    cfg = _config(kv_transfer_config=kv_transfer_config)

    assert cuda_gqa_cp_is_prefill_producer(cfg) is expected


@pytest.mark.parametrize(
    "num_tokens,world_size",
    [
        (16, 4),
        (10, 4),
        (1, 8),
        (4096, 8),
    ],
)
def test_gqa_cp_even_token_gather_reconstructs_identity(num_tokens, world_size):
    x = torch.randn(num_tokens, 11)

    shards = [local_token_shard(x, world_size, rank) for rank in range(world_size)]
    reconstructed = gather_first_dim_token_shards(shards, num_tokens)

    assert reconstructed.shape == x.shape
    torch.testing.assert_close(reconstructed, x, rtol=0, atol=0)
