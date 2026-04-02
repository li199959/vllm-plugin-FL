# Copyright (c) 2025 BAAI. All rights reserved.

import logging
from typing import Optional, List

from vllm.model_executor.custom_op import CustomOp
from .layernorm import *  # noqa F403 F401
from .activation import *  # noqa F403 F401
from .rotary_embedding import *  # noqa F403 F401

_HAS_FUSED_MOE = True
try:
    from .fused_moe import *  # noqa F403 F401
except Exception as e:  # pragma: no cover - import-time compatibility guard
    _HAS_FUSED_MOE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import FL fused_moe ops, falling back to vLLM native MoE: %s",
        e,
    )

logger = logging.getLogger(__name__)

FusedMoEFL = globals().get("FusedMoEFL") if _HAS_FUSED_MOE else None
UnquantizedFusedMoEMethodFL = (
    globals().get("UnquantizedFusedMoEMethodFL") if _HAS_FUSED_MOE else None
)

# Build OOT_OPS dynamically to avoid referencing None values
_OOT_OP_DEFS: list[tuple[str, object, str]] = [
    ("silu_and_mul", SiluAndMulFL, "SiluAndMul"),  # noqa F405
    ("gelu_and_mul", GeluAndMulFL, "GeluAndMul"),  # noqa F405
    ("rms_norm", RMSNormFL, "RMSNorm"),  # noqa F405
    ("rotary_embedding", RotaryEmbeddingFL, "RotaryEmbedding"),  # noqa F405
]
if FusedMoEFL is not None:
    _OOT_OP_DEFS.append(("fused_moe", FusedMoEFL, "FusedMoE"))  # noqa F405
if UnquantizedFusedMoEMethodFL is not None:
    _OOT_OP_DEFS.append((  # noqa F405
        "unquantized_fused_moe_method",
        UnquantizedFusedMoEMethodFL,
        "UnquantizedFusedMoEMethod",
    ))

OOT_OPS: dict[str, tuple[object, str]] = {k: (cls, name) for k, cls, name in _OOT_OP_DEFS}


def register_oot_ops(whitelist: Optional[List[str]] = None) -> None:
    """
    Register OOT (out-of-tree) custom operators.

    Args:
        whitelist: If provided, only register operators in this list.
                   If None, check VLLM_FL_OOT_WHITELIST env var.
                   If neither is set, register all operators.

    Operators in VLLM_FL_OOT_BLACKLIST or platform config oot_blacklist
    will be excluded from registration.
    """
    from vllm_fl.utils import (
        get_oot_blacklist,
        get_oot_whitelist,
        is_oot_enabled,
        use_flaggems_op,
    )
    from vllm.platforms import current_platform

    # Check if OOT registration is enabled
    if not is_oot_enabled():
        return

    # Get blacklist (from env var or platform config)
    blacklist = get_oot_blacklist() or []

    # Determine which operators to register
    env_whitelist = get_oot_whitelist()
    if env_whitelist is not None:
        ops_to_register = env_whitelist
    elif whitelist is not None:
        ops_to_register = whitelist
    else:
        ops_to_register = list(OOT_OPS.keys())

    # Be conservative for dense decoder models on vendor backends:
    # rotary affects q/k directly and a signature mismatch across vLLM versions
    # can corrupt generation without raising obvious runtime errors.
    # Keep the official rotary path unless the user explicitly whitelists it.
    explicit_whitelist = env_whitelist is not None or whitelist is not None
    if current_platform.device_type == "cuda" and getattr(current_platform, "vendor_name", "") == "metax":
        if not explicit_whitelist and "rotary_embedding" in ops_to_register:
            logger.warning(
                "Skipping OOT op 'rotary_embedding' on metax by default; "
                "use VLLM_FL_OOT_WHITELIST=rotary_embedding to force-enable it."
            )
            ops_to_register = [op for op in ops_to_register if op != "rotary_embedding"]

    # Apply blacklist
    ops_to_register = [op for op in ops_to_register if op not in blacklist]

    for op_name in ops_to_register:
        if op_name not in OOT_OPS:
            logger.warning(f"OOT op '{op_name}' not found in OOT_OPS, skipping.")
            continue

        # unquantized_fused_moe_method only registers when use_flaggems_op is True
        if op_name == "unquantized_fused_moe_method" and not use_flaggems_op(op_name):
            logger.debug(f"Skipping '{op_name}': use_flaggems_op returned False")
            continue

        op_cls, registration_name = OOT_OPS[op_name]
        logger.info(f"Registering oot op: {op_name} as '{registration_name}'")
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=registration_name)

    # Register attention backend only when explicitly requested.
    # Attention backend replacement is a high-risk change for text generation.
    if str(__import__("os").environ.get("VLLM_FL_ENABLE_CUSTOM_ATTENTION", "0")).lower() in ("1", "true"):
        try:
            from vllm_fl.dispatch.backends.flaggems.impl.custom_attention import register_attention
            logger.info("Registering attention backend: AttentionFLBackend")
            register_attention()
        except Exception as e:
            logger.warning(f"Failed to register attention backend: {e}")

    # Apply Ascend NPU monkey-patches if running on NPU.
    # These replace upstream module-level functions (e.g. in qwen3_next) with
    # Ascend implementations that bypass the CustomOp/dispatch path.
    from vllm.platforms import current_platform
    if current_platform.device_type == "npu":
        from vllm_fl.dispatch.backends.vendor.ascend.patch import apply_ascend_patches
        apply_ascend_patches()
