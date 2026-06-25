# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from types import SimpleNamespace


def _install_fake_triton_attn(monkeypatch):
    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    triton_attn = types.ModuleType("vllm.v1.attention.backends.triton_attn")

    class TritonAttentionImpl:
        supports_pcp = False

        def __init__(self):
            self.pcp_world_size = 2

        def forward(self, value):
            return value + 1

    triton_attn.TritonAttentionImpl = TritonAttentionImpl
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.backends.triton_attn",
        triton_attn,
    )
    return TritonAttentionImpl


def _install_fake_flash_attn(monkeypatch):
    flash_attn = types.ModuleType("vllm.v1.attention.backends.flash_attn")

    class FlashAttentionImpl:
        supports_pcp = False

        def __init__(self):
            self.pcp_world_size = 2

        def forward(self, value):
            return value + 10

    flash_attn.FlashAttentionImpl = FlashAttentionImpl
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.backends.flash_attn",
        flash_attn,
    )
    return FlashAttentionImpl


def test_qwen35_pcp_triton_bringup_patch_disabled(monkeypatch):
    TritonAttentionImpl = _install_fake_triton_attn(monkeypatch)
    FlashAttentionImpl = _install_fake_flash_attn(monkeypatch)
    monkeypatch.delenv("VLLM_FL_ENABLE_EXPERIMENTAL_QWEN35_PCP_TRITON", raising=False)

    module = importlib.reload(
        importlib.import_module("vllm_fl.patches.qwen35_pcp_triton_bringup")
    )
    module.apply_qwen35_pcp_triton_bringup_patch()

    assert TritonAttentionImpl.supports_pcp is False
    assert not hasattr(TritonAttentionImpl, "_fl_qwen35_pcp_attention_patched")
    assert FlashAttentionImpl.supports_pcp is False
    assert not hasattr(FlashAttentionImpl, "_fl_qwen35_pcp_attention_patched")


def test_qwen35_pcp_triton_bringup_patch_enabled(monkeypatch):
    TritonAttentionImpl = _install_fake_triton_attn(monkeypatch)
    FlashAttentionImpl = _install_fake_flash_attn(monkeypatch)
    monkeypatch.setenv("VLLM_FL_ENABLE_EXPERIMENTAL_QWEN35_PCP_TRITON", "1")

    module = importlib.reload(
        importlib.import_module("vllm_fl.patches.qwen35_pcp_triton_bringup")
    )
    module.apply_qwen35_pcp_triton_bringup_patch()

    assert TritonAttentionImpl.supports_pcp is True
    assert TritonAttentionImpl._fl_qwen35_pcp_attention_patched is True
    assert TritonAttentionImpl().forward(1) == 2
    assert FlashAttentionImpl.supports_pcp is True
    assert FlashAttentionImpl._fl_qwen35_pcp_attention_patched is True
    assert FlashAttentionImpl().forward(1) == 11


def test_qwen35_pcp_hybrid_kv_cache_patch_handles_hybrid_cp(monkeypatch):
    monkeypatch.setenv("VLLM_FL_ENABLE_EXPERIMENTAL_QWEN35_PCP_TRITON", "1")
    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.engine",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")

    def original_resolve(_kv_cache_config, _vllm_config):
        raise ValueError(
            "Hybrid KV cache groups with multiple block sizes do not "
            "support context parallelism (dcp_world_size/pcp_world_size > 1)."
        )

    kv_cache_utils.resolve_kv_cache_block_sizes = original_resolve
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.core.kv_cache_utils",
        kv_cache_utils,
    )
    sys.modules["vllm.v1.core"].kv_cache_utils = kv_cache_utils

    engine_core = types.ModuleType("vllm.v1.engine.core")
    engine_core.resolve_kv_cache_block_sizes = original_resolve
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", engine_core)

    module = importlib.reload(
        importlib.import_module("vllm_fl.patches.qwen35_pcp_triton_bringup")
    )
    module.apply_qwen35_pcp_hybrid_kv_cache_patch()

    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=256)),
        ],
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=2,
        ),
    )

    assert kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    ) == (512, 512)
    assert engine_core.resolve_kv_cache_block_sizes(kv_cache_config, vllm_config) == (
        512,
        512,
    )
