#!/usr/bin/env python3
"""Sweep Gems fused_experts_impl tiles for Qwen3-VL-30B-A3B MoE (TP2 shard).

Shapes: E=128, hidden 2048, intermediate 768/2 per rank, top-8, BF16.
Gems has no embedded tile table for PPU, so it uses get_default_config.
Each candidate shares BLOCK_SIZE_M between the align step and both GEMMs;
the GEMM1/GEMM2 N/K tiles, warps and stages are swept independently.
Prints the default time and the best candidates per M; --json writes the
winning table keyed by M.

  CUDA_VISIBLE_DEVICES=14 python3 sweep_moe_tiles.py --rows 64 128 --json out.json
"""

import argparse
import json
import sys

import torch

E, HIDDEN, INTER, TOPK = 128, 2048, 384, 8


def _routing(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(m, E, device="cuda", generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    return (w / w.sum(-1, keepdim=True)).float(), ids.to(torch.int32)


def _time(fn, iters=20):
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def _tiles(m):
    bms = (16, 32, 64) if m <= 256 else (32, 64, 128)
    g1 = [
        (bn, bk, w, s)
        for bn in (32, 64, 128)
        for bk in (64, 128, 256)
        for w in (4, 8)
        for s in (2, 3, 4)
        if bn * bk <= 128 * 256
    ]
    g2 = [
        (bn, bk, w, s)
        for bn in (32, 64, 128, 256)
        for bk in (64, 128)
        for w in (4, 8)
        for s in (2, 3, 4)
        if bn * bk <= 256 * 128
    ]
    return bms, g1, g2


def _cfg(bm, tile, group_m=1):
    bn, bk, w, s = tile
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": group_m,
        "num_warps": w,
        "num_stages": s,
    }


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args()

    import flag_gems  # noqa: F401

    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune

    _disable_flaggems_runtime_autotune()
    fm = sys.modules["flag_gems.fused.fused_moe"]
    default_lookup = fm.try_get_optimal_moe_config
    torch.manual_seed(0)
    w1 = (torch.randn(E, 2 * INTER, HIDDEN, device="cuda") * 0.02).bfloat16()
    w2 = (torch.randn(E, HIDDEN, INTER, device="cuda") * 0.02).bfloat16()

    def install(g1, g2):
        def lookup(*args, gemm_stage="gemm1", return_is_embedded=False, **kw):
            cfg = dict(g1 if gemm_stage == "gemm1" else g2)
            return (cfg, False) if return_is_embedded else cfg

        fm.try_get_optimal_moe_config = lookup if g1 else default_lookup

    def run(m, x, routes):
        def fn():
            for tw, ti in routes:
                fm.fused_experts_impl(x, w1, w2, tw, ti)

        return _time(fn) / len(routes)

    table = {}
    for m in a.rows:
        x = (torch.randn(m, HIDDEN, device="cuda") * 0.5).bfloat16()
        routes = [_routing(m, s) for s in range(a.seeds)]
        install(None, None)
        ref = fm.fused_experts_impl(x, w1, w2, *routes[0])
        base = run(m, x, routes)
        bms, g1s, g2s = _tiles(m)
        best = (base, None, None)
        for bm in bms:
            # GEMM1 sweep with a neutral GEMM2, then GEMM2 with the best GEMM1.
            g2_seed = _cfg(bm, (64, 64, 4, 3))
            g1_best = None
            for t in g1s:
                c1 = _cfg(bm, t)
                try:
                    install(c1, g2_seed)
                    us = run(m, x, routes)
                except Exception:  # noqa: BLE001 - invalid tile (smem/regs)
                    continue
                if g1_best is None or us < g1_best[0]:
                    g1_best = (us, c1)
            if g1_best is None:
                continue
            for t in g2s:
                c2 = _cfg(bm, t)
                try:
                    install(g1_best[1], c2)
                    us = run(m, x, routes)
                except Exception:  # noqa: BLE001
                    continue
                if us < best[0]:
                    best = (us, g1_best[1], c2)
        if best[1] is not None:
            install(best[1], best[2])
            out = fm.fused_experts_impl(x, w1, w2, *routes[0])
            err = (out.float() - ref.float()).abs().max().item()
            table[m] = {
                "gemm1": best[1],
                "gemm2": best[2],
                "us": best[0],
                "default_us": base,
            }
        else:
            err = 0.0
        install(None, None)
        print(
            f"M={m:5d} default {base:8.1f} us  best {best[0]:8.1f} us "
            f"({base / best[0]:.2f}x) max|diff| {err:.2e}  g1={best[1]} g2={best[2]}",
            flush=True,
        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
