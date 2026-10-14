# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in, model-specific optimizations grouped by hardware vendor.

Layout: ``optimizations/<vendor>/<model>/``. Each model package exposes
``enabled()`` (reads its own env switches), ``register()`` (installs kernels
and patches; idempotent) and optionally ``moe_experts()`` (returns an experts
class or None). Shared plugin code calls only the two hooks below, so adding a
model never touches ops/ or fused_moe/.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


def _model_packages():
    from vllm_fl.optimizations import ppu

    return [m for m in ppu.model_packages() if m.enabled()]


def _vendor_ok(vendor: str) -> bool:
    from vllm.platforms import current_platform

    actual = getattr(current_platform, "vendor_name", None)
    if actual != vendor:
        logger.warning_once(
            "Skipping %s-specific optimizations on vendor %r", vendor, actual
        )
        return False
    return True


def register_model_optimizations() -> None:
    """Install every enabled model optimization for the current vendor."""
    for model in _model_packages():
        if _vendor_ok(model.VENDOR):
            model.register()


def select_moe_experts():
    """Return a model-specific experts class, or None to keep the default."""
    for model in _model_packages():
        experts = getattr(model, "moe_experts", None)
        if experts is not None and _vendor_ok(model.VENDOR):
            cls = experts()
            if cls is not None:
                return cls
    return None
