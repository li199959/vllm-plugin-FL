# SPDX-License-Identifier: Apache-2.0
"""Opt-in weight packing for the Qwen dense 27B BF16 TP2 down projection."""
import functools
import logging
import os

logger = logging.getLogger(__name__)


def apply_dense_weight_layout_patch():
    if os.getenv("VLLM_FL_DENSE_DOWN_LAYOUT", "0") != "1":
        return
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    original = UnquantizedLinearMethod.process_weights_after_loading
    if getattr(original, "_fl_dense_down_layout", False):
        return

    @functools.wraps(original)
    def process(self, layer):
        original(self, layer)
        weight = getattr(layer, "weight", None)
        if (
            weight is None
            or weight.device.type != "cuda"
            or weight.dtype != torch.bfloat16
            or weight.ndim != 2
            or tuple(weight.shape) != (5120, 8704)
            or not getattr(layer, "prefix", "").endswith(".mlp.down_proj")
            or getattr(layer, "tp_size", None) != 2
            or weight.stride() != (8704, 1)
        ):
            return
        # F.linear uses weight.T. Store that operand contiguously once at
        # load time. Shape, BF16 values, and the Parameter object are retained.
        # No duplicate weight is kept and no forward-time copy is introduced.
        with torch.no_grad():
            weight.data = weight.data.t().contiguous().t()
        logger.info("[FL-DENSE-LAYOUT] packed %s shape=%s stride=%s",
                    layer.prefix, tuple(weight.shape), weight.stride())

    process._fl_dense_down_layout = True
    UnquantizedLinearMethod.process_weights_after_loading = process
