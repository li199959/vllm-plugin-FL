# Copyright (c) 2025 BAAI. All rights reserved.

import logging
import os

from . import version as version  # PyTorch-style: vllm_fl.version.git_version
from vllm_fl.utils import get_op_config as _get_op_config

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
        cfg.ALLOWED_LAYER_TYPES = getattr(cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ())


def _register_flagcx_connector():
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for _alias in ("FlagCXConnector", "FlagcxConnector"):
        if _alias not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                _alias,
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            )


def register():
    """Register the FL platform."""
    _patch_transformers_compat()

    # Platform-only patches must not import model/runtime modules here.  This
    # entrypoint is called while vLLM is still selecting the platform/vendor
    # backend; importing GDN model code here can make vendor detection fall back
    # to "unknown" and unintentionally bypass FlagGems.
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform

    glm5_platform()

    # Note: FlagCX connector registration is deferred to register_model()
    # to avoid circular imports during VllmConfig.__post_init__ in spawned
    # subprocesses.

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()

    return "vllm_fl.platform.PlatformFL"


def register_quant_linear():
    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel

    add_oot_quant_kernel()


def register_router():
    from vllm_fl.ops.fused_moe.router import replace_router_with_fl

    replace_router_with_fl()


def _apply_gdn_model_patches():
    """Apply optional GDN runtime patches after platform/vendor selection."""
    from vllm_fl.patches.gdn_mixed_prefill_decode import (
        apply_gdn_mixed_prefill_decode_patch,
    )

    apply_gdn_mixed_prefill_decode_patch()
    from vllm_fl.patches.gdn_fused_projection import (
        apply_gdn_fused_projection_patch,
    )

    apply_gdn_fused_projection_patch()


def register_model():
    """Register FL-specific models not yet upstream."""
    _register_flagcx_connector()
    _apply_gdn_model_patches()

    # Register OOT quant kernels so kernel selection can find them
    register_quant_linear()
    register_router()

    # Register GLM-5 (GlmMoeDsa) config not yet upstream
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY

        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig

        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        # from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        # glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
