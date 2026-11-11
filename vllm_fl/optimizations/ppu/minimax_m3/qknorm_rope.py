# Copyright (c) 2026 BAAI. All rights reserved.
"""Route M3's fused Q/K norm + RoPE + KV insert to Triton (VLLM_FL_M3_FUSED_IMPL=triton).

The kernel itself lives in ``vllm_fl.ops.minimax_m3_triton`` (it is also the
generic fallback when the native extension is absent). This override applies
even when ``vllm_fl._m3_fused_C`` registered the CUDA implementation.
"""

import torch

_library = None


def register_triton_qknorm_rope() -> None:
    global _library
    if _library is not None:
        return
    from vllm_fl.ops import minimax_m3_ops
    from vllm_fl.ops.minimax_m3_triton import (
        triton_fused_minimax_m3_qknorm_rope_kv_insert,
    )

    minimax_m3_ops.register_minimax_m3_ops()
    fallback_lib = getattr(minimax_m3_ops.register_minimax_m3_ops, "_lib", None)
    if fallback_lib is not None:
        # PyTorch cannot replace a Python CUDA impl with a second Python
        # impl. The fallback already dispatches through _TRITON_MODE.
        minimax_m3_ops._TRITON_MODE = True
        _library = fallback_lib
        return

    lib = torch.library.Library("_C", "IMPL")
    lib.impl(
        "fused_minimax_m3_qknorm_rope_kv_insert",
        triton_fused_minimax_m3_qknorm_rope_kv_insert,
        "CUDA",
    )
    _library = lib
