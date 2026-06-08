# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_fl.patches.qwen_dcp import (
    apply_qwen_dcp_config,
    choose_qwen_dcp_size,
    is_qwen_dcp_model,
    should_use_qwen_dcp_attention_backend,
)


def _vllm_config(
    *,
    model_type: str = "qwen3_5_moe_text",
    architectures: list[str] | None = None,
    tp_size: int = 16,
    dcp_size: int = 1,
    q_heads: int = 64,
    kv_heads: int = 8,
    additional_config: dict | None = None,
):
    text_config = SimpleNamespace(
        model_type=model_type,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
        architectures=architectures or [],
    )
    return SimpleNamespace(
        additional_config=additional_config or {},
        model_config=SimpleNamespace(
            hf_text_config=text_config,
            hf_config=text_config,
            architectures=architectures or [],
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            decode_context_parallel_size=dcp_size,
            dcp_comm_backend="ag_rs",
            cp_kv_cache_interleave_size=1,
        ),
    )


def test_qwen_dcp_model_detection_by_model_type():
    cfg = _vllm_config()

    assert is_qwen_dcp_model(cfg.model_config)


def test_qwen_dcp_model_detection_by_architecture():
    cfg = _vllm_config(
        model_type="unknown",
        architectures=["Qwen3NextForCausalLM"],
    )

    assert is_qwen_dcp_model(cfg.model_config)


def test_choose_qwen_dcp_size_auto_uses_largest_valid_size():
    cfg = _vllm_config(tp_size=32, q_heads=64, kv_heads=8)

    assert choose_qwen_dcp_size(cfg, "auto") == 4


def test_choose_qwen_dcp_size_rejects_invalid_gqa_layout():
    cfg = _vllm_config(tp_size=8, q_heads=64, kv_heads=8)

    with pytest.raises(ValueError, match="Invalid Qwen DCP configuration"):
        choose_qwen_dcp_size(cfg, 2)


def test_apply_qwen_dcp_config_from_additional_config(monkeypatch):
    monkeypatch.delenv("VLLM_FL_FORCE_FL_ATTENTION", raising=False)
    cfg = _vllm_config(
        tp_size=16,
        q_heads=64,
        kv_heads=8,
        additional_config={
            "enable_qwen_dcp": True,
            "qwen_dcp_size": "auto",
            "qwen_dcp_comm_backend": "a2a",
            "qwen_dcp_interleave_size": 2,
        },
    )

    apply_qwen_dcp_config(cfg)

    assert cfg.parallel_config.decode_context_parallel_size == 2
    assert cfg.parallel_config.dcp_comm_backend == "a2a"
    assert cfg.parallel_config.cp_kv_cache_interleave_size == 2
    assert should_use_qwen_dcp_attention_backend()

