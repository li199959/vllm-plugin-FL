#!/usr/bin/env python3
"""Time one attention layer on 16k/1k-style mixed batches: Triton vs PPU FA3.

Batch = one chunked-prefill sequence (q_len new tokens on top of a cached
prefix) plus `decodes` decode sequences, paged BF16 KV (block 16), Qwen3-VL
TP2 heads (16 q / 2 kv / 128). Prints per-call time and max |diff|.

  CUDA_VISIBLE_DEVICES=15 python3 bench_prefill_attention.py
"""

import argparse

import torch

HQ, HKV, D, BS = 16, 2, 128, 16


def _time(fn, iters=10):
    for _ in range(2):
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
    ap.add_argument(
        "--cases", default="8192:8192:63,8192:0:0,4096:12288:63,2048:2048:63"
    )
    ap.add_argument("--decode-ctx", type=int, default=16384)
    a = ap.parse_args()

    import flash_attn_3._C  # noqa: F401  (registers torch.ops.flash_attn_3)

    import vllm.v1.attention.ops.triton_unified_attention as ua

    def fa3_fwd(q, k, v, out, cu_q, max_q, seqused, table, splits=0):
        # Same call as vllm_fl's thead backend (whose module no longer imports
        # on vLLM 0.24); PPU FA3 wants max_seqlen_k=1 for paged KV.
        return torch.ops.flash_attn_3.fwd(
            q,
            k,
            v,
            None,
            None,
            None,
            out,
            cu_q,
            None,
            None,
            None,
            seqused,
            max_q,
            1,
            table,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            D**-0.5,
            True,
            -1,
            -1,
            0,
            0.0,
            True,
            None,
            splits,
            None,
            0,
            None,
        )

    for case in a.cases.split(","):
        q_len, prefix, decodes = map(int, case.split(":"))
        q_lens = [q_len] + [1] * decodes
        k_lens = [prefix + q_len] + [a.decode_ctx] * decodes
        per = [(n + BS - 1) // BS for n in k_lens]
        nblocks = sum(per) + 4
        k = (torch.randn(nblocks, BS, HKV, D, device="cuda") * 0.5).bfloat16()
        v = (torch.randn(nblocks, BS, HKV, D, device="cuda") * 0.5).bfloat16()
        table = torch.zeros(len(q_lens), max(per), dtype=torch.int32)
        perm = torch.randperm(nblocks - 4, dtype=torch.int32)
        off = 0
        for i, n in enumerate(per):
            table[i, :n] = perm[off : off + n]
            off += n
        table = table.cuda()
        q = (torch.randn(sum(q_lens), HQ, D, device="cuda") * 0.5).bfloat16()
        cu_q = torch.tensor(
            [0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device="cuda"
        )
        seqused = torch.tensor(k_lens, dtype=torch.int32, device="cuda")
        out_t, out_f = torch.empty_like(q), torch.empty_like(q)

        def triton():
            ua.unified_attention(
                q,
                k,
                v,
                out_t,
                cu_q,
                max(q_lens),
                seqused,
                max(k_lens),
                D**-0.5,
                True,
                (-1, -1),
                table,
                0,
                None,
                None,
                None,
            )

        def fa3(splits=0):
            fa3_fwd(q, k, v, out_f, cu_q, max(q_lens), seqused, table, splits)

        t_us, f_us = _time(triton), _time(fa3)
        err = (out_t.float() - out_f.float()).abs().max().item()
        s1_us = _time(lambda: fa3(1))
        err1 = (out_t.float() - out_f.float()).abs().max().item()
        print(
            f"q={q_len:5d} prefix={prefix:5d} +{decodes:2d} decode: triton {t_us / 1000:8.2f} ms  "
            f"fa3 {f_us / 1000:8.2f} ms  fa3(num_splits=1) {s1_us / 1000:8.2f} ms  "
            f"max|diff| {err:.2e} / {err1:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
