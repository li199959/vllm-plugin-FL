"""Exercise source-only registration in a fresh dispatcher."""

import os
import subprocess
import sys
from pathlib import Path


def test_triton_registration_without_native_extension():
    root = Path(__file__).resolve().parents[5]
    code = """
import sys
import torch
# Force the source-only path, even when a developer has built an extension.
sys.modules['vllm._C'] = None
sys.modules['vllm_fl._m3_fused_C'] = None
from vllm_fl.ops import minimax_m3_ops as ops
from vllm_fl.optimizations.ppu.minimax_m3.qknorm_rope import register_triton_qknorm_rope
ops.register_minimax_m3_ops()
register_triton_qknorm_rope()
register_triton_qknorm_rope()
assert ops._TRITON_MODE is True
assert torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert.default._schema
qkv = torch.empty((0, 768), device='cuda', dtype=torch.bfloat16)
w = torch.zeros(128, device='cuda', dtype=torch.bfloat16)
cs = torch.empty((1, 64), device='cuda', dtype=torch.bfloat16)
pos = torch.empty(0, device='cuda', dtype=torch.int64)
torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(
    qkv, w, w, cs, pos, 4, 1, 64, 1e-6, None, None, 0,
    None, None, None, None, 0, None, None, 'auto')
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
