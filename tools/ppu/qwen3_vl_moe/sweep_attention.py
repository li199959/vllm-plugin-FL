#!/usr/bin/env python3
"""Sweep vLLM Triton unified_attention decode launch params for Qwen3-VL (TP2).

Per rank: 16 query heads, 2 KV heads, head 128, BF16 paged KV cache. Decode
batches <= seq_threshold_3D (64 here) use the 3D split-softmax kernel plus
reduce_segments; larger batches use the 2D kernel. Swept: decode KV tile,
num_warps, num_stages and (3D only) the number of softmax segments. Each
candidate is checked against the default output.

  CUDA_VISIBLE_DEVICES=15 python3 sweep_attention.py --batches 32 64 128
"""

import argparse
import itertools
import sys

import torch

HQ, HKV, D = 16, 2, 128


class _LaunchOverride:
    """Wraps kernel_unified_attention so ``[grid](...)`` gets extra launch kwargs."""

    def __init__(self, kernel):
        self.kernel, self.extra = kernel, {}

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        return lambda *a, **kw: launch(*a, **{**kw, **self.extra})


def _time(fn, iters=50):
    for _ in range(5):
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
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    ap.add_argument("--ctx", type=int, nargs=2, default=[4096, 5120])
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--threshold-3d", type=int, default=64)
    ap.add_argument("--segs", type=int, nargs="+", default=[8, 16, 32])
    a = ap.parse_args()

    import vllm.v1.attention.ops.triton_unified_attention as ua

    override = _LaunchOverride(ua.kernel_unified_attention)
    ua.kernel_unified_attention = override
    default_tile = ua._get_tile_size
    bs = a.block_size
    torch.manual_seed(0)
    for b in a.batches:
        lens = torch.randint(a.ctx[0], a.ctx[1] + 1, (b,), dtype=torch.int32)
        blocks_per = (lens + bs - 1) // bs
        nblocks = int(blocks_per.sum()) + 8
        k = (torch.randn(nblocks, bs, HKV, D, device="cuda") * 0.5).bfloat16()
        v = (torch.randn(nblocks, bs, HKV, D, device="cuda") * 0.5).bfloat16()
        width = int(blocks_per.max())
        table = torch.zeros(b, width, dtype=torch.int32)
        perm = torch.randperm(nblocks - 8, dtype=torch.int32)
        off = 0
        for i, n in enumerate(blocks_per.tolist()):
            table[i, :n] = perm[off : off + n]
            off += n
        table = table.cuda()
        seqused = lens.cuda()
        q = (torch.randn(b, HQ, D, device="cuda") * 0.5).bfloat16()
        cu_q = torch.arange(b + 1, dtype=torch.int32, device="cuda")
        use_3d = b <= a.threshold_3d

        def call(out, segments):
            bufs = {}
            if use_3d:
                bufs = dict(
                    seq_threshold_3D=a.threshold_3d,
                    num_par_softmax_segments=segments,
                    softmax_segm_output=torch.empty(
                        a.threshold_3d, HQ, segments, D, device="cuda"
                    ),
                    softmax_segm_max=torch.empty(
                        a.threshold_3d, HQ, segments, device="cuda"
                    ),
                    softmax_segm_expsum=torch.empty(
                        a.threshold_3d, HQ, segments, device="cuda"
                    ),
                )
            return lambda: ua.unified_attention(
                q,
                k,
                v,
                out,
                cu_q,
                1,
                seqused,
                int(lens.max()),
                D**-0.5,
                True,
                (-1, -1),
                table,
                0,
                None,
                None,
                None,
                **bufs,
            )

        ref = torch.empty_like(q)
        override.extra = {}
        ua._get_tile_size = default_tile
        base = _time(call(ref, 16))
        best = (base, "default")
        tiles = (16, 32, 64)
        segs = tuple(a.segs) if use_3d else (16,)
        for tile, warps, stages, seg in itertools.product(
            tiles, (2, 4, 8), (1, 2, 3), segs
        ):
            # Non-TD path resolves the physical block per KV slot, so tiles larger
            # than block_size are legal; the output check guards correctness.
            ua._get_tile_size = lambda *args, _t=tile, **kw: _t
            override.extra = {"num_warps": warps, "num_stages": stages}
            out = torch.empty_like(q)
            try:
                fn = call(out, seg)
                us = _time(fn)
            except Exception:  # noqa: BLE001 - invalid launch config
                continue
            err = (out.float() - ref.float()).abs().max().item()
            if err > 2e-2:
                print(
                    f"  B={b} tile={tile} w={warps} s={stages} seg={seg} mismatch {err:.3e}"
                )
                continue
            if us < best[0]:
                best = (us, f"tile={tile} warps={warps} stages={stages} seg={seg}")
        ua._get_tile_size = default_tile
        override.extra = {}
        print(
            f"B={b:4d} {'3D' if use_3d else '2D'} default {base:8.1f} us  best {best[0]:8.1f} us "
            f"({base / best[0]:.2f}x) {best[1]}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
