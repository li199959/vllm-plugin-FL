# Copyright (c) 2026 BAAI. All rights reserved.
"""Sweep bf16 linear tiles for the Qwen3-VL-Embedding-8B shapes on one PPU.

LLM (TP1, no bias): qkv 4096->6144, o 4096->4096, gate_up 4096->24576,
down 12288->4096. Compares cuBLAS (torch) with the Gems linear_kernel at
explicit tiles; prints the best tile and the Gems/cuBLAS time ratio per (shape, M).

  CUDA_VISIBLE_DEVICES=<gpu> python3 sweep_linear.py [--m 64,256,...] [--out tiles.json]
"""

import argparse
import itertools
import json
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from vllm_fl.optimizations.ppu.minimax_m3.linear import _gems_linear  # noqa: E402

# name -> (K, N)
SHAPES = {
    "qkv": (4096, 6144),
    "o": (4096, 4096),
    "gate_up": (4096, 24576),
    "down": (12288, 4096),
}
TILES = [
    (bm, bn, bk, w, s)
    for bm, bn, bk in itertools.product((64, 128), (64, 128, 256), (32, 64))
    for w, s in ((4, 3), (4, 4), (8, 3), (8, 4))
    if not (bm == 128 and bn == 256 and bk == 64)
]


def bench(fn):
    return triton.testing.do_bench(fn, warmup=5, rep=25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="64,256,1024,2048,4096,8192")
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.manual_seed(0)
    result = {}
    for name, (k, n) in SHAPES.items():
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        for m in map(int, a.m.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = torch.nn.functional.linear(x, w)
            t_cublas = bench(lambda: torch.nn.functional.linear(x, w))
            best = None
            for tile in TILES:
                try:
                    out = _gems_linear(x, w, None, tile)
                    if not torch.allclose(
                        out.float(), ref.float(), atol=0.5, rtol=0.05
                    ):
                        continue
                    t = bench(lambda: _gems_linear(x, w, None, tile))
                except Exception:
                    continue
                if best is None or t < best[0]:
                    best = (t, tile)
            ratio = best[0] / t_cublas if best else float("inf")
            print(
                f"{name:8s} M={m:5d} cuBLAS {t_cublas * 1e3:8.1f} us  Gems best {best[0] * 1e3:8.1f} us "
                f"x{ratio:4.2f}  tile {best[1]}",
                flush=True,
            )
            result.setdefault(name, {})[m] = {
                "cublas_us": t_cublas * 1e3,
                "gems_us": best[0] * 1e3,
                "ratio": ratio,
                "tile": best[1],
            }
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
