# Copyright (c) 2025 BAAI. All rights reserved.

import logging
from typing import Optional, List

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from .layernorm import *  # noqa F403 F401
from .activation import *  # noqa F403 F401
from .rotary_embedding import *  # noqa F403 F401
from .fused_moe import *  # noqa F403 F401

# vllm logger (not stdlib) so INFO registration logs are actually emitted.
logger = init_logger(__name__)
_ = logging.getLogger(__name__ + ".unused")

# Mapping from OOT operator name (op_name, internal/whitelist) to (class, registration_name).
# registration_name is passed to CustomOp.register_oot and must match what vLLM uses
# when looking up the OOT op (typically the base class name).
# item example as follows:
# op_name: (class, registration_name of vllm's CustomOp.register_oot)
# note: cannot control inner gems op of UnquantizedFusedMoEMethodFL via env variable.
OOT_OPS = {
    "silu_and_mul": (SiluAndMulFL, "SiluAndMul"),  # noqa F405
    "gelu_and_mul": (GeluAndMulFL, "GeluAndMul"),  # noqa F405
    "rms_norm": (RMSNormFL, "RMSNorm"),  # noqa F405
    "rotary_embedding": (RotaryEmbeddingFL, "RotaryEmbedding"),  # noqa F405
    # NOTE: fused_moe is NOT registered via PluggableLayer/CustomOp.register_oot.
    # In vllm >= 0.24.0, FusedMoE is a factory function (not a class), so the
    # PluggableLayer OOT path is incompatible.  Instead, FusedMoEFL is injected
    # via monkey-patch in register_oot_ops() below.
    # "fused_moe": (FusedMoEFL, "FusedMoE"),
    # unquantized_fused_moe_method is also handled via FusedMoEFL factory —
    # no separate registration needed.
    # "unquantized_fused_moe_method": (UnquantizedFusedMoEMethodFL, "UnquantizedFusedMoEMethod"),
}

def _disable_flaggems_runtime_autotune() -> None:
    """Pin flag_gems LibTuner to its first config (no runtime benchmarking).

    The REPLAY benchmark protocol captures cudagraphs at first-seen shapes
    mid-serving; under heavy mixed load it hit an illegal memory access
    (flagtune do_bench_cudagraph replay) and killed the engine. Skipping the
    bench keeps serving deterministic; opt back in with
    VLLM_FL_FLAGGEMS_AUTOTUNE=1 (tuning should then be done offline).
    """
    import os

    if os.environ.get("VLLM_FL_FLAGGEMS_AUTOTUNE", "0") == "1":
        return
    try:
        from flag_gems.utils.libentry import LibTuner

        if getattr(LibTuner, "_fl_no_autotune", False):
            return

        def _policy_first_config(self, fn, configs, args, kwargs):
            cfg = next(iter(configs), None)
            if cfg is None:
                cfg = self.configs[0]
            return cfg, {}

        LibTuner.policy = _policy_first_config
        # Newer Gems registers policy subclasses which override the base
        # method. Generated index kernels instantiate these during serving;
        # patch the registered implementations as well as the base class.
        for tuner in LibTuner._dispatch_table.values():
            tuner.policy = _policy_first_config
        LibTuner._fl_no_autotune = True
        logger.info(
            "FlagGems runtime autotune disabled (first-config pinned); "
            "set VLLM_FL_FLAGGEMS_AUTOTUNE=1 to re-enable."
        )
    except Exception as e:
        logger.warning("Failed to disable FlagGems autotune: %s", e)


def _apply_minimax_m3_eager_break() -> None:
    """Wrap the MiniMax-M3 fused op with eager_break_during_capture.

    Deferred to worker init: vllm.compilation.breakable_cudagraph cannot be
    imported during early plugin registration (circular import), and the
    pure-torch fallback must stay out of breakable-cudagraph captures.
    """
    try:
        from vllm_fl.patches.minimax_m3 import _apply_fused_op_eager_break

        _apply_fused_op_eager_break()
    except Exception as e:
        logger.warning("M3 fused-op eager-break wrap failed: %s", e)


def _patch_unquantized_moe_oracle() -> None:
    """
    Monkey-patch the upstream select_unquantized_moe_backend so it does not
    short-circuit to (OOT, None) on our platform.  Instead it falls through
    to the normal CUDA/ROCm backend priority selection — the same logic that
    select_unquantized_moe_backend_oot uses.

    This is needed when FusedMoEFL is NOT registered (PREFER_ENABLED=0 or
    fused_moe blacklisted): without the patch, the in-tree UnquantizedFusedMoEMethod
    would get (OOT, None), skip _setup_kernel, and crash at inference time.
    """
    import vllm.model_executor.layers.fused_moe.oracle.unquantized as _oracle_mod
    from vllm_fl.ops.fused_moe.fused_moe_utils import select_unquantized_moe_backend_oot
    _oracle_mod.select_unquantized_moe_backend = select_unquantized_moe_backend_oot
    # Also patch the import in unquantized_fused_moe_method module
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as _method_mod
    _method_mod.select_unquantized_moe_backend = select_unquantized_moe_backend_oot
    logger.info("Patched select_unquantized_moe_backend to bypass OOT short-circuit")


def register_oot_ops(whitelist: Optional[List[str]] = None) -> None:
    """
    Register OOT (out-of-tree) custom operators.

    Args:
        whitelist: If provided, only register operators in this list.
                   If None, check VLLM_FL_OOT_WHITELIST env var.
                   If neither is set, register all operators.

    Operators in VLLM_FL_OOT_BLACKLIST or platform config oot_blacklist
    will be excluded from registration.

    When fused_moe is not registered (PREFER_ENABLED=0 or blacklisted),
    the upstream select_unquantized_moe_backend oracle is monkey-patched
    so it picks native CUDA backends instead of returning (OOT, None).
    """
    from vllm_fl.utils import get_oot_blacklist, get_oot_whitelist, is_oot_enabled, use_flaggems_op

    # Activate the native M3 fused kernel (registers _C::fused_minimax_m3_*)
    # before anything registers the python fallback.
    try:
        import vllm_fl._m3_fused_C  # noqa: F401

        logger.info_once("Native M3 fused kernel active (vllm_fl._m3_fused_C)")
    except (ImportError, OSError) as e:
        logger.debug("vllm_fl._m3_fused_C unavailable: %s", e)

    # Must run before flag_gems.enable() below: pin LibTuner before any
    # kernel gets its first (tuning) call.
    _disable_flaggems_runtime_autotune()

    # Opt-in per-model/per-vendor kernels (vllm_fl/optimizations/); must run
    # after native schemas are registered above.
    from vllm_fl.optimizations import register_model_optimizations

    register_model_optimizations()

    # Deferred M3 fused-op eager-break wrap: breakable_cudagraph is not
    # importable during early plugin load, so bind it once workers are up.
    _apply_minimax_m3_eager_break()

    # Check if OOT registration is enabled
    if not is_oot_enabled():
        # Patch the upstream oracle so in-tree FusedMoE works on this platform.
        _patch_unquantized_moe_oracle()
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

    # Apply blacklist
    ops_to_register = [op for op in ops_to_register if op not in blacklist]

    # If fused_moe is excluded (blacklisted or not in whitelist), patch the
    # upstream oracle so the in-tree FusedMoE doesn't crash on OOT platforms.
    if "fused_moe" not in ops_to_register:
        _patch_unquantized_moe_oracle()

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
        if issubclass(op_cls, PluggableLayer):
            PluggableLayer.register_oot(_decorated_layer_cls=op_cls, name=registration_name)
        else:
            CustomOp.register_oot(_decorated_op_cls=op_cls, name=registration_name)
        # Apply Ascend NPU monkey-patches if running on NPU.
        # These replace upstream module-level functions (e.g. in qwen3_next) with
        # Ascend implementations that bypass the CustomOp/dispatch path.
        from vllm.platforms import current_platform
        if current_platform.device_type == "npu":
            from vllm_fl.dispatch.backends.vendor.ascend.patch import apply_ascend_patches
            apply_ascend_patches()

        # Apply Sunrise/PTPU monkey-patches if running on PTPU.
        if current_platform.device_type == "ptpu":
            from vllm_fl.dispatch.backends.vendor.sunrise.patch import apply_sunrise_patches
            apply_sunrise_patches()

    # --- FusedMoE monkey-patch (vllm >= 0.24.0) ---
    # FusedMoE is a factory function in vllm 0.24.0+, not a PluggableLayer
    # subclass, so it cannot be registered via CustomOp/PluggableLayer.register_oot.
    # Instead we replace the factory function in the two places vllm imports it
    # from, so all model code transparently gets FusedMoEFL.
    if "fused_moe" not in (blacklist or []):
        _patch_fused_moe_factory()


def _patch_fused_moe_factory() -> None:
    """Replace the FusedMoE factory function with FusedMoEFL in all relevant
    vllm modules so that model code picks up the FL version automatically."""
    import inspect
    import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
    import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer

    if getattr(_fused_moe_layer, "FusedMoE", None) is FusedMoEFL:  # noqa F405
        # Already patched — idempotent.
        return

    # Patch at the module level so `from vllm...fused_moe import FusedMoE` picks it up.
    _fused_moe_layer.FusedMoE = FusedMoEFL  # noqa F405
    _fused_moe_pkg.FusedMoE = FusedMoEFL   # noqa F405
    logger.info("Monkey-patched FusedMoE factory -> FusedMoEFL")
