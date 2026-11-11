#!/usr/bin/env python3
"""Decode attention per layer: tuned Triton 3D/2D vs PPU FA3, by context length.

Qwen3-VL TP2 heads (16 q / 2 kv / 128), paged BF16 KV (block 16), one query
token per sequence. Triton uses vLLM's split-softmax 3D path for
B <= threshold (with this package's tuned launch) and 2D above it.

  CUDA_VISIBLE_DEVICES=2 python3 bench_decode_attention.py --ctx 1024 4096 16384
"""

import argparse

import torch

HQ, HKV, D, BS = 16, 2, 128, 16


def _time(fn, iters=30):
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
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--batches", type=int, nargs="+", default=[64])
    ap.add_argument("--threshold-3d", type=int, default=64)
    ap.add_argument(
        "--splits",
        type=int,
        nargs="*",
        default=[1],
        help="fixed FA3 num_splits values to time as well",
    )
    a = ap.parse_args()

    import vllm.v1.attention.ops.triton_unified_attention as ua

    from vllm_fl.optimizations.ppu.qwen3_vl_moe import attention

    attention.install_tuned_attention()
    import flash_attn_3._C  # noqa: F401

    th = a.threshold_3d
    segs = 16
    bufs = dict(
        seq_threshold_3D=th,
        num_par_softmax_segments=segs,
        softmax_segm_output=torch.empty(th, HQ, segs, D, device="cuda"),
        softmax_segm_max=torch.empty(th, HQ, segs, device="cuda"),
        softmax_segm_expsum=torch.empty(th, HQ, segs, device="cuda"),
    )
    for b in a.batches:
        for ctx in a.ctx:
            lens = torch.full((b,), ctx, dtype=torch.int32)
            lens -= torch.arange(b, dtype=torch.int32) * (
                ctx // (4 * b)
            )  # spread lengths
            per = (lens + BS - 1) // BS
            nblocks = int(per.sum()) + 4
            k = (torch.randn(nblocks, BS, HKV, D, device="cuda") * 0.5).bfloat16()
            v = (torch.randn(nblocks, BS, HKV, D, device="cuda") * 0.5).bfloat16()
            table = torch.zeros(b, int(per.max()), dtype=torch.int32)
            perm = torch.randperm(nblocks - 4, dtype=torch.int32)
            off = 0
            for i, n in enumerate(per.tolist()):
                table[i, :n] = perm[off : off + n]
                off += n
            table, seqused = table.cuda(), lens.cuda()
            q = (torch.randn(b, HQ, D, device="cuda") * 0.5).bfloat16()
            cu_q = torch.arange(b + 1, dtype=torch.int32, device="cuda")
            out_t, out_f = torch.empty_like(q), torch.empty_like(q)
            max_k = int(lens.max())

            def triton():
                ua.unified_attention(
                    q,
                    k,
                    v,
                    out_t,
                    cu_q,
                    1,
                    seqused,
                    max_k,
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

            def fa3(splits=0):
                attention.fa3_paged_attention(
                    q,
                    k,
                    v,
                    out_f,
                    cu_q,
                    1,
                    seqused,
                    table,
                    D**-0.5,
                    num_splits=splits,
                )

            t_us, f_us = _time(triton), _time(fa3)
            err = (out_t.float() - out_f.float()).abs().max().item()
            fixed = "  ".join(f"s{n}={_time(lambda n=n: fa3(n)):.1f}" for n in a.splits)
            print(
                f"B={b:4d} ctx={ctx:6d} triton {t_us:8.1f} us  fa3(auto) {f_us:8.1f} us  "
                f"fixed: {fixed}  max|diff| {err:.2e}",
                flush=True,
            )
            del k, v


if __name__ == "__main__":
    main()
