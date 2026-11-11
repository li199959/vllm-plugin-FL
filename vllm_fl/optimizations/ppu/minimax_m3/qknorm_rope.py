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
    from vllm_fl.ops.minimax_m3_triton import (
        triton_fused_minimax_m3_qknorm_rope_kv_insert,
    )

    lib = torch.library.Library("_C", "IMPL")
    lib.impl(
        "fused_minimax_m3_qknorm_rope_kv_insert",
        triton_fused_minimax_m3_qknorm_rope_kv_insert,
        "CUDA",
    )
    _library = lib
