#!/usr/bin/env python3
"""Decode attention as vLLM serves it: FULL CUDA graph captured with dummy
lengths, block_table width = max_model_len / block_size, replayed with real
lengths. Compares tuned Triton (3D) and PPU FA3 per layer call.

  CUDA_VISIBLE_DEVICES=3 python3 bench_decode_graph.py --ctx 1024 4096 --width 512
"""

import argparse

import torch

HQ, HKV, D, BS = 16, 2, 128, 16


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--width", type=int, default=512, help="block_table columns")
    ap.add_argument(
        "--layers", type=int, default=48, help="calls per graph (distinct KV)"
    )
    a = ap.parse_args()
    import vllm.v1.attention.ops.triton_unified_attention as ua

    from vllm_fl.optimizations.ppu.qwen3_vl_moe import attention

    attention.install_tuned_attention()
    import flash_attn_3._C  # noqa: F401

    b, w = a.batch, a.width
    nblocks = b * w + 4
    # One shared pool sized for the full table, like the KV cache; each "layer"
    # gets its own pool so reads are cold as in serving.
    pools = [
        (torch.randn(nblocks, BS, HKV, D, device="cuda") * 0.5).bfloat16()
        for _ in range(min(a.layers, 8))
    ]
    table = torch.arange(b * w, dtype=torch.int32, device="cuda").view(b, w)
    q = (torch.randn(b, HQ, D, device="cuda") * 0.5).bfloat16()
    cu = torch.arange(b + 1, dtype=torch.int32, device="cuda")
    seqused = torch.ones(b, dtype=torch.int32, device="cuda")  # capture-time dummy
    out = torch.empty_like(q)
    th, segs = 64, 16
    bufs = dict(
        seq_threshold_3D=th,
        num_par_softmax_segments=segs,
        softmax_segm_output=torch.empty(th, HQ, segs, D, device="cuda"),
        softmax_segm_max=torch.empty(th, HQ, segs, device="cuda"),
        softmax_segm_expsum=torch.empty(th, HQ, segs, device="cuda"),
    )
    max_k = w * BS

    def triton(kv):
        ua.unified_attention(
            q,
            kv,
            kv,
            out,
            cu,
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

    def fa3(kv):
        attention.fa3_paged_attention(q, kv, kv, out, cu, 1, seqused, table, D**-0.5)

    graphs = {}
    for name, fn in (("triton", triton), ("fa3", fa3)):
        for i in range(a.layers):
            fn(pools[i % len(pools)])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(a.layers):
                fn(pools[i % len(pools)])
        graphs[name] = g
    for ctx in a.ctx:
        seqused.copy_(
            torch.full((b,), ctx, dtype=torch.int32)
            - torch.arange(b, dtype=torch.int32) * (ctx // (4 * b))
        )
        res = {}
        for name, g in graphs.items():
            g.replay()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(5):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            res[name] = s.elapsed_time(e) / (5 * a.layers) * 1000
        print(
            f"B={b} ctx={ctx} width={w} graph replay per call: triton {res['triton']:.1f} us  "
            f"fa3 {res['fa3']:.1f} us",
            flush=True,
        )


if __name__ == "__main__":
    main()
