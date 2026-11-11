#!/usr/bin/env python3
"""Decode-size dense GEMMs for Qwen3-VL (TP2), cold weights: Gems mm vs native vs
Gems linear_kernel with swept tiles.

In serving every layer's MoE reads ~600 MB of expert weights, so dense weights
are never L2-resident; each call here rotates through enough weight copies to
exceed L2. Shapes: qkv 2048->2560, o_proj 2048->2048.

  CUDA_VISIBLE_DEVICES=14 python3 sweep_decode_dense.py --rows 1 16 32 64 128
"""

import argparse

import torch
import triton

SHAPES = {"qkv": (2048, 2560), "o_proj": (2048, 2048), "router": (2048, 128)}
COPIES = 24  # x ~10 MB per weight > L2


def _time(fn, replays=5):
    # Serving runs these inside CUDA graphs, so time graph replay (GPU time
    # only, no Python launch overhead) over COPIES distinct cold weights.
    for i in range(COPIES):
        fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(COPIES):
            fn(i)
    g.replay()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(replays):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (replays * COPIES) * 1000


def _gems_linear(x, w, tile, out):
    from flag_gems.ops.linear import linear_kernel

    m, k = x.shape
    n = w.shape[0]
    bm, bn, bk, warps, stages = tile
    kernel = linear_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x,
        w,
        w,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        n,
        1,
        0,
        False,
        bm,
        bn,
        bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 16, 32, 64, 128])
    a = ap.parse_args()
    import flag_gems

    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune

    _disable_flaggems_runtime_autotune()
    native = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    tiles = [
        (bm, bn, bk, w, s)
        for bm in (16, 32, 64)
        for bn in (32, 64, 128)
        for bk in (64, 128, 256)
        for w in (4, 8)
        for s in (2, 3, 4)
        if bn * bk <= 128 * 256
    ]
    for name, (k, n) in SHAPES.items():
        ws = [
            (torch.randn(n, k, device="cuda") * 0.02).bfloat16() for _ in range(COPIES)
        ]
        for m in a.rows:
            x = torch.randn(m, k, device="cuda").bfloat16()
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            ref = torch.ops.aten.mm.default.redispatch(native, x, ws[0].t())
            gems_us = _time(lambda i: flag_gems.mm(x, ws[i].t()))
            nat_us = _time(
                lambda i: torch.ops.aten.mm.default.redispatch(native, x, ws[i].t())
            )
            best = (float("inf"), None)
            for tile in tiles:
                if tile[0] > max(16, triton.next_power_of_2(m)):
                    continue
                try:
                    us = _time(lambda i, t=tile: _gems_linear(x, ws[i], t, out))
                except Exception:  # noqa: BLE001 - invalid tile
                    continue
                if us < best[0]:
                    best = (us, tile)
            _gems_linear(x, ws[0], best[1], out)
            err = (out.float() - ref.float()).abs().max().item()
            roof = (n * k * 2) / 2.1e12 * 1e6
            print(
                f"{name:6s} M={m:4d} gems_mm {gems_us:6.1f}  native {nat_us:6.1f}  "
                f"tuned_linear {best[0]:6.1f} us {best[1]}  (roofline {roof:.1f})  "
                f"err {err:.1e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
