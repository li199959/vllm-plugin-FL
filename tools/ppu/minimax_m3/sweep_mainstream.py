"""Sweep the decode main-stream ops the hybrid owns (cold inputs).

--op norm    : Gems fused add RMSNorm num_warps vs flashinfer Gemma kernel
--op qkv     : split-K bf16 [M,6144]x[1024,6144]
--op router  : split-K fp32 [M,6144]x[128,6144] (3xTF32)
--op oproj   : Gems linear [M,512]x[6144,512] tiles vs split-K
Prints the best config per M next to the current choice.
"""

import argparse
import contextlib
import json

import torch
import triton
from sweep_small_m import measure


def ring(shape, dtype, nbytes=1 << 30, scale=0.02):
    count = max(8, nbytes // (torch.Size(shape).numel() * dtype.itemsize))
    return [
        torch.randn(shape, device="cuda", dtype=dtype) * scale for _ in range(count)
    ]


def best_of(cands, run, copies):
    out = []
    for cfg in cands:
        with contextlib.suppress(Exception):  # resource limits
            out.append((measure(lambda i: run(cfg, i), copies), cfg))
    out.sort(key=lambda r: r[0])
    return out


@torch.inference_mode()
def norm(rows):
    from flag_gems.fused.fused_add_rms_norm import fused_add_rms_norm_kernel

    width = 6144
    w = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1
    shifted = w.float() + 1.0
    for m in rows:
        xs = ring((m, width), torch.bfloat16, 1 << 28, 1.0)
        rs = ring((m, width), torch.bfloat16, 1 << 28, 1.0)
        copies = len(xs)

        def gems(warps, i):
            fused_add_rms_norm_kernel[(m,)](
                xs[i],
                rs[i],
                shifted,
                xs[i].stride(0),
                1,
                rs[i].stride(0),
                1,
                width,
                1e-6,
                triton.next_power_of_2(width),
                num_warps=warps,
            )

        res = best_of([4, 8, 16, 32], gems, copies)
        row = {"op": "norm", "m": m, "gems": [(round(t, 2), c) for t, c in res]}
        try:
            import flashinfer

            row["flashinfer_gemma"] = round(
                measure(
                    lambda i: flashinfer.norm.gemma_fused_add_rmsnorm(
                        xs[i], rs[i], w, 1e-6
                    ),
                    copies,
                ),
                2,
            )
        except Exception as e:  # noqa: BLE001
            row["flashinfer_gemma"] = str(e)[:60]
        yield row


def splitk_sweep(op, rows, n, dtype):
    from vllm_fl.optimizations.ppu.minimax_m3.splitk import (
        splitk_config,
        splitk_linear_impl,
    )

    k = 6144
    ws = ring((n, k), dtype)
    copies = len(ws)
    for m in rows:
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        native = measure(lambda i: torch.nn.functional.linear(x, ws[i]), copies)
        cur = splitk_config(m, n, k, dtype)
        current = measure(lambda i: splitk_linear_impl(x, ws[i], None, cur), copies)
        bms = [16] if m <= 16 else ([32] if m <= 32 else [32, 64])
        bks = (32, 64, 128) if dtype == torch.float32 else (64, 128, 256)
        cands = [
            (bm, bn, bk, sp, w, st)
            for bm in bms
            for bn in (32, 64, 128)
            for bk in bks
            for sp in (2, 4, 8, 16, 32)
            for w in (4, 8)
            for st in (2, 3, 4)
            if k // sp >= bk
        ]
        res = best_of(cands, lambda c, i: splitk_linear_impl(x, ws[i], None, c), copies)
        yield {
            "op": op,
            "m": m,
            "native": round(native, 2),
            "current": round(current, 2),
            "current_cfg": list(cur),
            "best": [(round(t, 2), list(c)) for t, c in res[:3]],
        }


@torch.inference_mode()
def oproj(rows):
    from flag_gems.ops.linear import linear_kernel

    from vllm_fl.optimizations.ppu.minimax_m3.linear import bf16_linear
    from vllm_fl.optimizations.ppu.minimax_m3.splitk import splitk_linear_impl

    raw = linear_kernel
    while not isinstance(raw, triton.runtime.JITFunction):
        raw = raw.fn
    k, n = 512, 6144
    ws = ring((n, k), torch.bfloat16)
    copies = len(ws)
    for m in rows:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        native = measure(lambda i: torch.nn.functional.linear(x, ws[i]), copies)
        current = measure(lambda i: bf16_linear(x, ws[i], None), copies)

        def gems(c, i):
            bm, bn, bk, w, st = c
            raw[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                x,
                ws[i],
                ws[i],
                out,
                m,
                n,
                k,
                k,
                1,
                k,
                1,
                n,
                1,
                0,
                False,
                bm,
                bn,
                bk,
                num_warps=w,
                num_stages=st,
            )

        cands = [
            (bm, bn, bk, w, st)
            for bm in (16, 32, 64)
            for bn in (32, 64, 128)
            for bk in (32, 64, 128)
            for w in (2, 4, 8)
            for st in (2, 3, 4)
            if bm <= max(16, m)
        ]
        g = best_of(cands, gems, copies)
        s = best_of(
            [
                (bm, bn, bk, sp, 4, st)
                for bm in (16, 32, 64)
                for bn in (32, 64, 128)
                for bk in (32, 64)
                for sp in (2, 4)
                for st in (2, 3)
                if bm <= max(16, m)
            ],
            lambda c, i: splitk_linear_impl(x, ws[i], None, c),
            copies,
        )
        yield {
            "op": "oproj",
            "m": m,
            "native": round(native, 2),
            "current": round(current, 2),
            "gems_best": [(round(t, 2), list(c)) for t, c in g[:3]],
            "splitk_best": [(round(t, 2), list(c)) for t, c in s[:2]],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op", choices=("norm", "qkv", "router", "oproj"), required=True
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args()
    torch.manual_seed(3)
    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune

    _disable_flaggems_runtime_autotune()
    with torch.inference_mode():
        gen = {
            "norm": lambda: norm(args.rows),
            "qkv": lambda: splitk_sweep("qkv", args.rows, 1024, torch.bfloat16),
            "router": lambda: splitk_sweep("router", args.rows, 128, torch.float32),
            "oproj": lambda: oproj(args.rows),
        }[args.op]()
        for row in gen:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
