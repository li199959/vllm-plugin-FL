#!/usr/bin/env python3
"""Per-layer dense GEMM time at prefill sizes: FlagGems vs native ATen (PPU).

Qwen3-VL-30B-A3B TP2 shapes: qkv 2048->2560, o_proj 2048->2048,
router 2048->128. Native is called through the CUDA dispatch key so that
an enabled FlagGems registration does not intercept it.

  CUDA_VISIBLE_DEVICES=15 python3 bench_prefill_dense.py --rows 2048 8192
"""

import argparse

import torch

SHAPES = {"qkv": (2048, 2560), "o_proj": (2048, 2048), "router": (2048, 128)}


def _time(fn, iters=20):
    for _ in range(3):
        fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    a = ap.parse_args()
    import flag_gems

    from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune

    _disable_flaggems_runtime_autotune()
    native = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    for m in a.rows:
        for name, (k, n) in SHAPES.items():
            x = torch.randn(m, k, device="cuda").bfloat16()
            w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
            gems_us = _time(lambda: flag_gems.mm(x, w.t()))
            nat_us = _time(
                lambda: torch.ops.aten.mm.default.redispatch(native, x, w.t())
            )
            print(
                f"M={m:5d} {name:7s} gems {gems_us:8.1f} us  native {nat_us:8.1f} us  "
                f"({gems_us / nat_us:.2f}x)",
                flush=True,
            )


if __name__ == "__main__":
    main()
