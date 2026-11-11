# Copyright (c) 2026 BAAI. All rights reserved.
"""Sweep fp16 linear tiles for the bge-m3 (XLM-R large) shapes on one PPU.

Shapes per layer: qkv 1024->3072, o 1024->1024, fc1 1024->4096, fc2 4096->1024,
all with bias. Compares cuBLAS (torch), the Gems default addmm path and the
Gems linear_kernel with explicit tiles; prints the best tile per (shape, M).

  CUDA_VISIBLE_DEVICES=3 python3 sweep_linear.py [--m 512,1024,...] [--out tiles.json]
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

SHAPES = {
    "qkv": (1024, 3072),
    "o": (1024, 1024),
    "fc1": (1024, 4096),
    "fc2": (4096, 1024),
}
TILES = [
    (bm, bn, bk, w, s)
    for bm, bn, bk in itertools.product((64, 128), (64, 128, 256), (32, 64))
    for w, s in ((4, 3), (4, 4), (8, 3), (8, 4))
    if not (bm == 128 and bn == 256 and bk == 64)
]


def bench(fn):
    return triton.testing.do_bench(fn, warmup=10, rep=40)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", default="512,1024,2048,4096,8192")
    ap.add_argument("--out")
    args = ap.parse_args()
    import flag_gems

    torch.manual_seed(0)
    result = {}
    for name, (k, n) in SHAPES.items():
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) / k**0.5
        b = torch.randn(n, device="cuda", dtype=torch.float16)
        for m in map(int, args.m.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            ref = torch.nn.functional.linear(x, w, b)
            t_cublas = bench(lambda: torch.nn.functional.linear(x, w, b))
            with flag_gems.use_gems():
                t_gems = bench(lambda: torch.nn.functional.linear(x, w, b))
            best = None
            for tile in TILES:
                try:
                    out = _gems_linear(x, w, b, tile)
                except Exception:  # tile does not fit shared memory
                    continue
                if not torch.allclose(out, ref, atol=2e-2, rtol=2e-2):
                    continue
                t = bench(lambda: _gems_linear(x, w, b, tile))
                if best is None or t < best[0]:
                    best = (t, tile)
            result[f"{name}:{m}"] = {
                "cublas": t_cublas,
                "gems": t_gems,
                "tuned": best[0],
                "tile": best[1],
            }
            print(
                f"{name:4s} M={m:5d} cublas {t_cublas * 1e3:7.1f}us gems {t_gems * 1e3:7.1f}us "
                f"tuned {best[0] * 1e3:7.1f}us {best[1]} ratio {best[0] / t_cublas:.2f}",
                flush=True,
            )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
