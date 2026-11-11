import json

import pytest

MODEL = "/mnt/cpfs/xiangbin/codex/Unlimited-OCR"


@pytest.fixture(scope="module")
def registered():
    import vllm_fl

    vllm_fl.register_model()


def _raw():
    try:
        return json.load(open(f"{MODEL}/config.json"))
    except FileNotFoundError:
        pytest.skip("Unlimited-OCR checkpoint not available")


def test_config_parses_backbone_and_window(registered):
    from vllm_fl.models.unlimited_ocr.config import UnlimitedOCRConfig

    cfg = UnlimitedOCRConfig(**_raw())
    text = cfg.text_config
    assert cfg.rswa_window == 128  # language_config.sliding_window_size
    assert text.kv_lora_rank is None  # plain MHA, not MLA
    # Transformers v5 derives head_dim = qk_rope_head_dim = 0 for this backbone;
    # vLLM's head size (and the Triton split-softmax buffers) must see 128.
    assert text.head_dim == text.hidden_size // text.num_attention_heads == 128


def test_model_config_head_size_and_registry(registered):
    from vllm.config import ModelConfig
    from vllm.model_executor.models.registry import ModelRegistry

    _raw()
    mc = ModelConfig(model=MODEL, trust_remote_code=True)
    assert mc.get_head_size() == 128
    assert not mc.use_mla
    assert "UnlimitedOCRForCausalLM" in ModelRegistry.get_supported_archs()


def test_tokenizer_decodes_spaces(registered):
    from vllm.tokenizers import get_tokenizer

    _raw()
    tok = get_tokenizer(MODEL, trust_remote_code=True)
    text = "Unlimited OCR Works: One-shot parsing.\n| Model | Score |"
    assert tok.decode(tok.encode(text, add_special_tokens=False)) == text
