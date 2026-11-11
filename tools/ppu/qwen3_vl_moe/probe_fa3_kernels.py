#!/usr/bin/env python3
"""Which FA3 kernels (forward variant, split-combine variant) each call shape launches.

The PPU FA3 wheel picks num_splits itself; each split-count bucket is a
different FlashAttnFwdCombine instantiation, so it changes the coverage
kernel set. Prints, per decode batch size (and a few prefill/mixed shapes),
the kernels seen under torch.profiler.

  CUDA_VISIBLE_DEVICES=10 python3 probe_fa3_kernels.py
"""

import argparse
import re

import torch

HQ, HKV, D, BS = 16, 2, 128, 16


def _short(name):
    if "FlashAttnFwdCombine" in name:
        m = re.search(r"FlashAttnFwdCombine<cute::tuple<.*?>\s*>,\s*(\d+),", name)
        return f"Combine<{m.group(1) if m else '?'}>"
    if "FlashAttnFwdSm80" in name:
        m = re.search(
            r"CollectiveMainloopFwdSm80<(\d+), (\d+), (\w+), cute::tuple<cute::C<(\d+)>, "
            r"cute::C<(\d+)>",
            name,
        )
        return f"Fwd<{','.join(m.groups())}>" if m else "Fwd<?>"
    if "prepare_varlen" in name:
        return "prepare_varlen"
    return None


def _batch(q_lens, k_lens):
    per = [(n + BS - 1) // BS for n in k_lens]
    nblocks = sum(per) + 4
    k = torch.zeros(nblocks, BS, HKV, D, device="cuda", dtype=torch.bfloat16)
    table = torch.zeros(len(q_lens), max(per), dtype=torch.int32)
    off = 0
    for i, n in enumerate(per):
        table[i, :n] = torch.arange(off, off + n, dtype=torch.int32)
        off += n
    q = torch.zeros(sum(q_lens), HQ, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    return (
        q,
        k,
        table.cuda(),
        torch.tensor(k_lens, dtype=torch.int32, device="cuda"),
        cu,
    )


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128],
    )
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 16384])
    ap.add_argument("--splits", type=int, default=0, help="num_splits passed to FA3")
    a = ap.parse_args()
    import flash_attn_3._C  # noqa: F401

    from vllm_fl.optimizations.ppu.qwen3_vl_moe import attention

    def probe(label, q_lens, k_lens, splits):
        q, k, table, lens, cu = _batch(q_lens, k_lens)
        out = torch.empty_like(q)
        run = lambda: attention.fa3_paged_attention(
            q, k, k, out, cu, max(q_lens), lens, table, D**-0.5, num_splits=splits
        )
        run()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            run()
            torch.cuda.synchronize()
        names = sorted({s for e in prof.events() if (s := _short(e.name))})
        print(f"{label:34s} splits={splits}: {' '.join(names)}", flush=True)

    for ctx in a.ctx:
        for b in a.batches:
            probe(f"decode B={b} ctx={ctx}", [1] * b, [ctx] * b, a.splits)
    for q_len, prefix, dec in (
        (8192, 8192, 63),
        (8192, 0, 63),
        (2048, 14336, 63),
        (8192, 0, 0),
        (300, 724, 40),
    ):
        for s in (0, 1):
            probe(
                f"mixed q={q_len} pre={prefix} +{dec}",
                [q_len] + [1] * dec,
                [prefix + q_len] + [16384] * dec,
                s,
            )


if __name__ == "__main__":
    main()
