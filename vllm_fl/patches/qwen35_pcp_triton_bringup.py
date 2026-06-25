# SPDX-License-Identifier: Apache-2.0
"""Experimental Qwen3.5/Qwen3.6 PCP bring-up patches.

vLLM 0.20.2 has the process-group and scheduler scaffolding for Prefill
Context Parallelism (PCP), but CUDA attention backends still advertise
``supports_pcp = False``.  That blocks Qwen3.5/Qwen3.6 hybrid-attention
experiments before the model reaches the real PCP implementation gaps.

This patch is intentionally narrow and disabled by default.  It only marks
selected attention implementations as PCP-capable so a server can move past
``check_attention_cp_compatibility`` and expose the next missing piece.

It is *not* a complete PCP attention implementation.  Correct full-attention
PCP still needs per-rank partial-attention LSE/output combination, or an
equivalent kernel/backend that computes attention over the complete context.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

_ENV_NAME = "VLLM_FL_ENABLE_EXPERIMENTAL_QWEN35_PCP_TRITON"
_PATCHED = False
_KV_CACHE_PATCHED = False
_FORWARD_WARNING_LOGGED = False
_HYBRID_KV_WARNING_LOGGED = False


def _enabled() -> bool:
    return os.environ.get(_ENV_NAME, "0").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def apply_qwen35_pcp_triton_bringup_patch() -> None:
    """Enable experimental attention-backend PCP bring-up hooks once."""
    global _PATCHED
    if _PATCHED or not _enabled():
        return

    backend_classes = []
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

        backend_classes.append(TritonAttentionImpl)
    except ImportError:
        logger.debug("TritonAttentionImpl is not importable; skip PCP gate patch")

    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

        backend_classes.append(FlashAttentionImpl)
    except ImportError:
        logger.debug("FlashAttentionImpl is not importable; skip PCP gate patch")

    if not backend_classes:
        _PATCHED = True
        return

    patched_names: list[str] = []
    for backend_cls in backend_classes:
        if getattr(backend_cls, "_fl_qwen35_pcp_attention_patched", False):
            continue

        if getattr(backend_cls, "supports_pcp", False):
            logger.info(
                "Skip experimental Qwen3.5 PCP attention patch for %s; "
                "backend already supports PCP",
                backend_cls.__name__,
            )
            continue

        def make_patched_forward(original_forward: Any) -> Any:
            def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
                global _FORWARD_WARNING_LOGGED
                if (
                    not _FORWARD_WARNING_LOGGED
                    and getattr(self, "pcp_world_size", 1) > 1
                ):
                    logger.warning(
                        "Experimental Qwen3.5/Qwen3.6 PCP attention bring-up "
                        "is enabled. This only bypasses the attention backend "
                        "supports_pcp gate; validate correctness before "
                        "benchmarking or production use."
                    )
                    _FORWARD_WARNING_LOGGED = True
                return original_forward(self, *args, **kwargs)

            return patched_forward

        patched_forward = make_patched_forward(backend_cls.forward)

        patched_forward._fl_qwen35_pcp_attention_patched = True  # type: ignore[attr-defined]
        backend_cls.forward = patched_forward
        backend_cls.supports_pcp = True
        backend_cls._fl_qwen35_pcp_attention_patched = True
        patched_names.append(backend_cls.__name__)

    _PATCHED = True

    if patched_names:
        logger.warning(
            "Applied experimental Qwen3.5/Qwen3.6 PCP attention gate patch "
            "to %s; set %s=0 to disable",
            patched_names,
            _ENV_NAME,
        )


def apply_qwen35_pcp_hybrid_kv_cache_patch() -> None:
    """Allow Qwen3.5/Qwen3.6 hybrid KV cache groups to proceed with PCP.

    vLLM 0.20.2 rejects context parallelism when a model has multiple KV cache
    group block sizes.  Qwen3.5/Qwen3.6 hybrid attention hits this because it
    has both full-attention KV cache and GDN/Mamba state cache groups.

    This experimental hook keeps the original behavior for all other cases.
    If the original resolver raises the known hybrid+CP error, it derives a
    scheduler/hash block size from the LCM of the per-group block sizes and the
    total CP world size.  This mirrors the single-group CP alignment rule and
    is sufficient to expose the next bring-up gap.
    """
    global _KV_CACHE_PATCHED
    if _KV_CACHE_PATCHED or not _enabled():
        return

    from vllm.v1.core import kv_cache_utils

    original_resolve = kv_cache_utils.resolve_kv_cache_block_sizes
    if getattr(original_resolve, "_fl_qwen35_pcp_hybrid_kv_patched", False):
        _KV_CACHE_PATCHED = True
        return

    def patched_resolve_kv_cache_block_sizes(
        kv_cache_config: Any,
        vllm_config: Any,
    ) -> tuple[int, int]:
        global _HYBRID_KV_WARNING_LOGGED
        try:
            return original_resolve(kv_cache_config, vllm_config)
        except ValueError as exc:
            if "Hybrid KV cache groups with multiple block sizes" not in str(exc):
                raise

            parallel_config = vllm_config.parallel_config
            dcp = parallel_config.decode_context_parallel_size
            pcp = parallel_config.prefill_context_parallel_size
            total_cp = dcp * pcp
            if total_cp <= 1:
                raise

            groups = kv_cache_config.kv_cache_groups
            group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
            scheduler_block_size = math.lcm(*group_block_sizes) * total_cp

            if not _HYBRID_KV_WARNING_LOGGED:
                logger.warning(
                    "Experimental Qwen3.5/Qwen3.6 PCP hybrid KV cache "
                    "bring-up is enabled. group_block_sizes=%s, total_cp=%s, "
                    "scheduler/hash_block_size=%s. Validate correctness before "
                    "benchmarking or production use.",
                    group_block_sizes,
                    total_cp,
                    scheduler_block_size,
                )
                _HYBRID_KV_WARNING_LOGGED = True

            # Use the scheduler block size as the hash block size. This keeps
            # prefix-hash granularity conservative and avoids invalid finer
            # hashes while the PCP hybrid path is still experimental.
            return scheduler_block_size, scheduler_block_size

    patched_resolve_kv_cache_block_sizes._fl_qwen35_pcp_hybrid_kv_patched = (  # type: ignore[attr-defined]
        True
    )
    kv_cache_utils.resolve_kv_cache_block_sizes = patched_resolve_kv_cache_block_sizes

    # vllm.v1.engine.core imports the resolver directly at module import time.
    # Patch that alias too if core.py is already loaded in this process.
    core_mod = sys.modules.get("vllm.v1.engine.core")
    if core_mod is not None:
        core_mod.resolve_kv_cache_block_sizes = patched_resolve_kv_cache_block_sizes

    _KV_CACHE_PATCHED = True

    logger.warning(
        "Applied experimental Qwen3.5/Qwen3.6 PCP hybrid KV cache patch; "
        "set %s=0 to disable",
        _ENV_NAME,
    )
