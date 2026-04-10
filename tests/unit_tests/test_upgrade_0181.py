import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm_fl
from vllm_fl.patches import glm_moe_dsa
from vllm_fl.platform import PlatformFL


class TestRegister0181:
    def test_register_returns_platform_and_sets_spawn(self, monkeypatch):
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

        patcher = MagicMock()
        get_op_config = MagicMock()
        monkeypatch.setattr(
            "vllm_fl.patches.glm_moe_dsa.apply_platform_patches",
            patcher,
        )
        monkeypatch.setattr(vllm_fl, "_get_op_config", get_op_config)

        result = vllm_fl.register()

        assert result == "vllm_fl.platform.PlatformFL"
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        patcher.assert_called_once_with()
        get_op_config.assert_called_once_with()


class TestPlatform0181:
    def test_metax_keeps_ragged_prefill_disabled(self, monkeypatch):
        pytest.importorskip("vllm", reason="vllm not installed")
        from vllm.config import CUDAGraphMode

        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                worker_cls=None,
                data_parallel_size=1,
            ),
            model_config=SimpleNamespace(
                use_mla=False,
                disable_cascade_attn=False,
            ),
            cache_config=SimpleNamespace(block_size=None),
            compilation_config=SimpleNamespace(
                compile_sizes=None,
                cudagraph_mode=CUDAGraphMode.NONE,
            ),
            attention_config=SimpleNamespace(
                use_cudnn_prefill=True,
                use_trtllm_ragged_deepseek_prefill=True,
                use_trtllm_attention=True,
                disable_flashinfer_prefill=False,
            ),
        )

        monkeypatch.setattr(PlatformFL, "vendor_name", "metax")
        monkeypatch.setattr(PlatformFL, "device_type", "cuda")

        PlatformFL.check_and_update_config(vllm_config)

        assert vllm_config.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"
        assert vllm_config.cache_config.block_size == 16
        assert vllm_config.model_config.disable_cascade_attn is True
        assert vllm_config.attention_config.use_cudnn_prefill is False
        assert vllm_config.attention_config.use_trtllm_ragged_deepseek_prefill is False
        assert vllm_config.attention_config.use_trtllm_attention is False
        assert vllm_config.attention_config.disable_flashinfer_prefill is True


class TestGlm0181Patches:
    def test_apply_model_patches_only_calls_remaining_patchers(self, monkeypatch):
        schedule_patch = MagicMock()
        rope_patch = MagicMock()
        monkeypatch.setattr(glm_moe_dsa, "patch_indexer_schedule_metadata", schedule_patch)
        monkeypatch.setattr(glm_moe_dsa, "patch_indexer_rope_reshape", rope_patch)

        glm_moe_dsa.apply_model_patches()

        schedule_patch.assert_called_once_with()
        rope_patch.assert_called_once_with()
        assert not hasattr(glm_moe_dsa, "patch_is_deepseek_mla")

    def test_patch_fp8_mqa_logits_dim_is_idempotent(self, monkeypatch):
        calls = []

        deep_gemm = ModuleType("vllm.utils.deep_gemm")

        def _orig_impl(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, *args, **kwargs):
            calls.append((kv[1], args, kwargs))
            return "ok"

        deep_gemm._fp8_mqa_logits_impl = _orig_impl
        deep_gemm._lazy_init = lambda: None
        monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", deep_gemm)

        glm_moe_dsa.patch_fp8_mqa_logits_dim()
        first_impl = deep_gemm._fp8_mqa_logits_impl
        glm_moe_dsa.patch_fp8_mqa_logits_dim()

        assert deep_gemm._fp8_mqa_logits_impl is first_impl
        assert getattr(first_impl, "_fl_glm_fp8_patch", False) is True

        scale = SimpleNamespace(flatten=lambda: "flattened-scale")
        result = first_impl("q", ("k", scale), "w", "ks", "ke", clean_logits=True)

        assert result == "ok"
        assert calls == [("flattened-scale", (), {"clean_logits": True})]

    def test_patch_indexer_schedule_metadata_is_idempotent(self, monkeypatch):
        builder_module = ModuleType("vllm.v1.attention.backends.mla.indexer")
        import_utils = ModuleType("vllm.utils.import_utils")
        deep_gemm = ModuleType("vllm.utils.deep_gemm")

        import_utils.has_deep_gemm = lambda: True
        deep_gemm.get_paged_mqa_logits_metadata = lambda seq_lens, block_size, num_sms: (
            "metadata",
            tuple(seq_lens),
            block_size,
            num_sms,
        )

        class FakeBuilder:
            def __init__(self):
                self.kv_cache_spec = SimpleNamespace(block_size=64)
                self.num_sms = 132
                self.scheduler_metadata_buffer = None

            def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
                self.scheduler_metadata_buffer = "old"
                return SimpleNamespace(
                    decode=SimpleNamespace(schedule_metadata="existing"),
                    num_decodes=2,
                )

        builder_module.DeepseekV32IndexerMetadataBuilder = FakeBuilder
        monkeypatch.setitem(sys.modules, "vllm.utils.import_utils", import_utils)
        monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", deep_gemm)
        monkeypatch.setitem(
            sys.modules,
            "vllm.v1.attention.backends.mla.indexer",
            builder_module,
        )

        glm_moe_dsa.patch_indexer_schedule_metadata()
        first_build = FakeBuilder.build
        glm_moe_dsa.patch_indexer_schedule_metadata()

        assert FakeBuilder.build is first_build
        assert getattr(first_build, "_fl_glm_schedule_patch", False) is True

        builder = FakeBuilder()
        common = SimpleNamespace(seq_lens=[11, 22, 33])
        result = builder.build(0, common)

        assert result.decode.schedule_metadata == "existing"
        assert builder.scheduler_metadata_buffer == ("metadata", (11, 22), 64, 132)
