# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.ops.dsa_cp import (
    build_token_plan,
    cuda_dsa_cp_enabled,
    cuda_dsa_cp_layer_sharding,
    gather_first_dim_token_shards,
    is_sparse_mla_model,
    local_token_shard,
    pad_first_dim,
    slice_first_dim_with_plan,
)


def _config(additional_config=None, hf_text_config=None, hf_config=None):
    return SimpleNamespace(
        additional_config=additional_config,
        model_config=SimpleNamespace(
            hf_text_config=hf_text_config,
            hf_config=hf_config,
        ),
    )


def test_build_token_plan_even_split():
    plan = build_token_plan(num_tokens=16, world_size=4, rank=2)

    assert plan.padded_num_tokens == 16
    assert plan.local_start == 8
    assert plan.local_end == 12
    assert plan.local_end_with_pad == 12
    assert plan.local_num_tokens == 4


def test_build_token_plan_pads_tail_rank():
    plan = build_token_plan(num_tokens=10, world_size=4, rank=3)

    assert plan.padded_num_tokens == 12
    assert plan.local_start == 9
    assert plan.local_end == 10
    assert plan.local_end_with_pad == 12
    assert plan.local_num_tokens == 1
    assert plan.local_num_tokens_with_pad == 3


def test_build_token_plan_rejects_invalid_rank():
    with pytest.raises(ValueError):
        build_token_plan(num_tokens=10, world_size=4, rank=4)


def test_cuda_dsa_cp_enabled_from_additional_config():
    cfg = _config(additional_config={"enable_dsa_cp": True})

    assert cuda_dsa_cp_enabled(cfg)


def test_cuda_dsa_cp_enabled_accepts_flashcomm_alias():
    cfg = _config(additional_config={"enable_flashcomm1": "1"})

    assert cuda_dsa_cp_enabled(cfg)


def test_cuda_dsa_cp_layer_sharding_normalizes_strings():
    cfg = _config(additional_config={"layer_sharding": ["q_b_proj", "o_proj"]})

    assert cuda_dsa_cp_layer_sharding(cfg) == ["q_b_proj", "o_proj"]


def test_sparse_mla_model_checks_index_topk():
    cfg = _config(hf_text_config=SimpleNamespace(index_topk=2048))

    assert is_sparse_mla_model(cfg)


def _simulate_token_parallel_gather(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reproduce the a_proj token-shard + all-gather + trim on a single host.

    This mirrors ``_fused_qkv_a_proj_token_parallel``: every rank slices its
    contiguous token block via the shared plan, right-pads to the common
    per-rank length, the padded shards are concatenated in rank order, and the
    padding tail is trimmed back to ``num_tokens``.
    """

    num_tokens = x.shape[0]
    padded_shards = []
    for rank in range(world_size):
        plan = build_token_plan(num_tokens, world_size, rank)
        shard = slice_first_dim_with_plan(x, plan)
        padded_shards.append(pad_first_dim(shard, plan.local_num_tokens_with_pad))
    return gather_first_dim_token_shards(padded_shards, num_tokens)


@pytest.mark.parametrize(
    "num_tokens,world_size",
    [
        (16, 4),  # evenly divisible
        (10, 4),  # tail rank partially filled
        (9, 8),   # later ranks fully empty (only padding)
        (1, 8),   # single token, most ranks empty
        (8, 1),   # no sharding
        (4096, 8),  # prefill-sized
    ],
)
def test_token_parallel_gather_reconstructs_identity(num_tokens, world_size):
    x = torch.randn(num_tokens, 7)

    reconstructed = _simulate_token_parallel_gather(x, world_size)

    assert reconstructed.shape == x.shape
    # Real token rows must be bit-identical; padding rows are trimmed away.
    torch.testing.assert_close(reconstructed, x, rtol=0, atol=0)


def _simulate_even_token_parallel_gather(
    x: torch.Tensor, world_size: int
) -> torch.Tensor:
    """Reproduce the a_proj even-shard + all-gather + trim on a single host.

    This mirrors ``_fused_qkv_a_proj_token_parallel`` after the cudagraph fix:
    every rank takes its *equal-length* contiguous block via ``local_token_shard``
    (input padded to a multiple of ``world_size``), the shards are concatenated
    in rank order, and the padding tail is trimmed back to ``num_tokens``.
    """

    num_tokens = x.shape[0]
    shards = [local_token_shard(x, world_size, rank) for rank in range(world_size)]
    return gather_first_dim_token_shards(shards, num_tokens)


@pytest.mark.parametrize(
    "num_tokens,world_size",
    [
        (16, 4),  # evenly divisible
        (10, 4),  # tail rank partially filled
        (9, 8),   # later ranks fully empty (only padding)
        (1, 8),   # single token: old ragged slice gave rank>=1 a negative length
        (8, 1),   # no sharding
        (4096, 8),  # prefill-sized
    ],
)
def test_even_token_parallel_gather_reconstructs_identity(num_tokens, world_size):
    x = torch.randn(num_tokens, 7)

    reconstructed = _simulate_even_token_parallel_gather(x, world_size)

    assert reconstructed.shape == x.shape
    torch.testing.assert_close(reconstructed, x, rtol=0, atol=0)


@pytest.mark.parametrize("world_size", [1, 4, 8])
def test_local_token_shard_is_equal_length_and_non_negative(world_size):
    # Regression: the old ragged slice ``hidden_states[start:min(end, n)]`` is
    # empty in eager but its symbolic length goes negative under torch.compile
    # (Inductor reinterpret_tensor numel overflow during cudagraph capture).
    # Even sharding must give every rank an identical, non-negative length.
    num_tokens = 1
    x = torch.randn(num_tokens, 7)
    expected_len = (num_tokens + world_size - 1) // world_size

    lengths = [
        local_token_shard(x, world_size, rank).shape[0] for rank in range(world_size)
    ]

    assert all(length == expected_len for length in lengths)
    assert all(length >= 0 for length in lengths)
