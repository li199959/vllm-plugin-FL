# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_fl.ops.dsa_cp import (
    build_token_plan,
    cuda_dsa_cp_enabled,
    cuda_dsa_cp_layer_sharding,
    is_sparse_mla_model,
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
