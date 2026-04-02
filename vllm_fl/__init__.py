# Copyright (c) 2025 BAAI. All rights reserved.

import os
import logging

from vllm_fl.utils import get_op_config as _get_op_config

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )

    # Fix: GLM/GLM-4/ChatGLM tokenizers cannot be converted from slow to fast
    # tokenizer format in transformers 4.57.6.
    #
    # Patch convert_slow_tokenizer at the module level. When vocab_file is None,
    # set it to "" to avoid crashes in tiktoken/bpe loaders.
    try:
        import transformers.convert_slow_tokenizer as _cst
        if not getattr(_cst.convert_slow_tokenizer, "_vllm_fl_patched", False):
            _orig = _cst.convert_slow_tokenizer

            def _safe_convert(tokenizer, *args, **kwargs):
                orig_vf = getattr(tokenizer, "vocab_file", None)
                if orig_vf is None:
                    tokenizer.vocab_file = ""
                try:
                    return _orig(tokenizer, *args, **kwargs)
                finally:
                    if orig_vf is None:
                        tokenizer.vocab_file = orig_vf

            _safe_convert._vllm_fl_patched = True
            _cst.convert_slow_tokenizer = _safe_convert

            try:
                import transformers.tokenization_utils_fast as _tuf
                _tuf.convert_slow_tokenizer = _safe_convert
            except Exception:
                pass
    except Exception:
        pass

    # Retry GLM-family tokenizer loading with use_fast=False when the
    # fast-tokenizer conversion path fails under transformers 4.57.x.
    try:
        import transformers.models.auto.tokenization_auto as _ta

        if not getattr(_ta.AutoTokenizer.from_pretrained,
                       "_vllm_fl_patched", False):
            _orig_from_pretrained = _ta.AutoTokenizer.from_pretrained.__func__

            def _should_retry_with_slow(exc: Exception) -> bool:
                err = str(exc)
                return any(msg in err for msg in (
                    "Converting from SentencePiece and Tiktoken failed",
                    "No such file or directory: ''",
                    "has no attribute truncation",
                ))

            def _safe_from_pretrained(cls, pretrained_model_name_or_path,
                                      *inputs, **kwargs):
                if kwargs.get("use_fast") is False:
                    return _orig_from_pretrained(
                        cls, pretrained_model_name_or_path, *inputs, **kwargs
                    )

                try:
                    return _orig_from_pretrained(
                        cls, pretrained_model_name_or_path, *inputs, **kwargs
                    )
                except (ValueError, FileNotFoundError, TypeError,
                        AttributeError) as exc:
                    if not _should_retry_with_slow(exc):
                        raise

                    retry_kwargs = dict(kwargs)
                    retry_kwargs["use_fast"] = False
                    logger.warning(
                        "Fast tokenizer conversion failed for %s; retrying "
                        "with use_fast=False. Original error: %s",
                        pretrained_model_name_or_path,
                        exc,
                    )
                    return _orig_from_pretrained(
                        cls, pretrained_model_name_or_path, *inputs,
                        **retry_kwargs
                    )

            _safe_from_pretrained._vllm_fl_patched = True
            _ta.AutoTokenizer.from_pretrained = classmethod(
                _safe_from_pretrained
            )
    except Exception:
        pass


def register():
    """Register the FL platform."""
    _patch_transformers_compat()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()
    return "vllm_fl.platform.PlatformFL"


def register_model():
    """Register the FL model."""
    from vllm import ModelRegistry
    import vllm.model_executor.models.qwen3_next as qwen3_next_module

    # Register Qwen3.5 MoE config
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.qwen3_5_moe import Qwen3_5MoeConfig
        _CONFIG_REGISTRY["qwen3_5_moe"] = Qwen3_5MoeConfig
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE config error: {str(e)}")

    # Register Qwen3Next model
    try:
        from vllm_fl.models.qwen3_next import Qwen3NextForCausalLM  # noqa: F401

        qwen3_next_module.Qwen3NextForCausalLM = Qwen3NextForCausalLM
        logger.warning(
            "Qwen3NextForCausalLM has been patched to use vllm_fl.models.qwen3_next, "
            "original vLLM implementation is overridden"
        )

        ModelRegistry.register_model(
            "Qwen3NextForCausalLM",
            "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register Qwen3Next model error: {str(e)}")

    # Register Qwen3.5 MoE model
    try:
        ModelRegistry.register_model(
            "Qwen3_5MoeForConditionalGeneration",
            "vllm_fl.models.qwen3_5:Qwen3_5MoeForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE model error: {str(e)}")

    # Register MiniCPMO model
    try:
        ModelRegistry.register_model(
            "MiniCPMO",
            "vllm_fl.models.minicpmo:MiniCPMO"
        )
    except Exception as e:
        logger.error(f"Register MiniCPMO model error: {str(e)}")

    # Register Kimi-K2.5 model
    try:
        ModelRegistry.register_model(
            "KimiK25ForConditionalGeneration",
            "vllm_fl.models.kimi_k25:KimiK25ForConditionalGeneration",
        )
    except Exception as e:
        logger.error(f"Register KimiK25 model error: {str(e)}")

    # Register GLM-5 (GlmMoeDsa) model
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        glm5_model()

        ModelRegistry.register_model(
            "GlmMoeDsaForCausalLM",
            "vllm_fl.models.glm_moe_dsa:GlmMoeDsaForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")

    # Register BGE-M3 pooling backport for vLLM 0.13.x
    try:
        ModelRegistry.register_model(
            "BgeM3EmbeddingModel",
            "vllm_fl.models.bge_m3:BgeM3EmbeddingModel",
        )
    except Exception as e:
        logger.error(f"Register BgeM3EmbeddingModel error: {str(e)}")
