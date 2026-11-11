# Copyright (c) 2026 BAAI. All rights reserved.
"""Register Unlimited-OCR (UnlimitedOCRForCausalLM) with vLLM v0.24.

Upstream vLLM supports the model from v0.25 (vllm-project/vllm#46564, #47102).
This installs the config class, config verifier and a lazy model path without
modifying the vLLM installation. Repeated calls are safe.
"""

import logging

logger = logging.getLogger(__name__)

_ARCH = "UnlimitedOCRForCausalLM"
_MODEL_TYPE = "unlimited_ocr"


def apply_unlimited_ocr_patches() -> bool:
    from vllm.model_executor.models import config as model_config
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import config as transformers_config

    from vllm_fl.models.unlimited_ocr.config import (
        UnlimitedOCRConfig,
        UnlimitedOCRForCausalLMConfig,
    )

    from vllm.tokenizers import registry as tokenizer_registry

    transformers_config._CONFIG_REGISTRY[_MODEL_TYPE] = UnlimitedOCRConfig
    # The checkpoint declares LlamaTokenizerFast, which under Transformers v5
    # decodes without spaces; load tokenizer.json with TokenizersBackend instead.
    tokenizer_registry._MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS.add(_MODEL_TYPE)
    model_config.MODELS_CONFIG_MAP[_ARCH] = UnlimitedOCRForCausalLMConfig
    model_registry._MULTIMODAL_MODELS.setdefault(_ARCH, ("deepseek_ocr", _ARCH))
    model_registry.ModelRegistry.register_model(
        _ARCH, f"vllm_fl.models.unlimited_ocr.model:{_ARCH}"
    )
    logger.info("Installed Unlimited-OCR model support")
    return True


__all__ = ["apply_unlimited_ocr_patches"]
