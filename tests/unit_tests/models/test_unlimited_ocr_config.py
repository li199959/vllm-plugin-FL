from types import SimpleNamespace

import pytest

from vllm_fl.models.unlimited_ocr.config import (
    UnlimitedOCRConfig,
    UnlimitedOCRForCausalLMConfig,
)


def test_config_parses_backbone_and_window():
    cfg = UnlimitedOCRConfig(
        language_config={
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "qk_nope_head_dim": 0,
            "qk_rope_head_dim": 0,
            "kv_lora_rank": None,
            "sliding_window_size": 128,
        }
    )
    assert cfg.rswa_window == 128
    assert cfg.text_config.kv_lora_rank is None
    assert cfg.text_config.head_dim == 128


@pytest.mark.parametrize("window", [64, 256])
def test_explicit_rswa_window_overrides_checkpoint(window):
    cfg = UnlimitedOCRConfig(
        rswa_window=window, language_config={"sliding_window_size": 128}
    )
    assert cfg.rswa_window == window


def test_verifier_selects_rswa_backend_and_disables_prefix_cache():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cfg = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )
    UnlimitedOCRForCausalLMConfig.verify_and_update_config(cfg)
    assert cfg.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert not cfg.cache_config.enable_prefix_caching
    UnlimitedOCRForCausalLMConfig.verify_and_update_config(cfg)


def test_verifier_rejects_backends_without_rswa():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cfg = SimpleNamespace(
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN)
    )
    with pytest.raises(ValueError, match="R-SWA is only implemented"):
        UnlimitedOCRForCausalLMConfig.verify_and_update_config(cfg)
