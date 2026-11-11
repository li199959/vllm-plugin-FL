"""Sweep the plugin split-K GEMM against native linear with cold weights."""

import argparse
import itertools
import json

import torch
from sweep_small_m import measure

SHAPES = [
    (6144, 128, torch.float32),
    (6144, 1024, torch.bfloat16),
    (6144, 384, torch.bfloat16),
]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256, 512]
    )
    args = parser.parse_args()
    torch.manual_seed(13)
    from vllm_fl.optimizations.ppu.minimax_m3.splitk import splitk_linear_impl

    for k, n, dtype in SHAPES:
        copies = max(8, (1 << 30) // (k * n * dtype.itemsize))
        ws = [
            torch.randn(n, k, device="cuda", dtype=dtype) * 0.02 for _ in range(copies)
        ]
        for m in args.rows:
            if dtype != torch.float32 and m > 128:
                continue
            x = torch.randn(m, k, device="cuda", dtype=dtype)
            native = measure(lambda i: torch.nn.functional.linear(x, ws[i]), copies)
            ref = torch.nn.functional.linear(x.double(), ws[0].double())
            best = None
            for bm, bn, bk, split, warps in itertools.product(
                (16, 32, 64), (32, 64), (32, 64), (2, 4, 8, 16, 32), (4,)
            ):
                if bm > max(16, m) * 2 or bm < min(64, m) // 2:
                    continue
                cfg = (bm, bn, bk, split, warps)
                try:
                    got = splitk_linear_impl(x, ws[0], None, cfg)
                    err = (got.double() - ref).abs().max().item()
                    us = measure(
                        lambda i: splitk_linear_impl(x, ws[i], None, cfg),
                        copies,
                    )
                except Exception:
                    continue
                if best is None or us < best[0]:
                    best = (us, list(cfg), err)
            native_err = (
                (torch.nn.functional.linear(x, ws[0]).double() - ref).abs().max().item()
            )
            print(
                json.dumps(
                    {
                        "k": k,
                        "n": n,
                        "dtype": str(dtype),
                        "m": m,
                        "native_us": native,
                        "best_us": best[0],
                        "best": best[1],
                        "ratio": best[0] / native,
                        "max_err": best[2],
                        "native_err": native_err,
                    }
                ),
                flush=True,
            )
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
