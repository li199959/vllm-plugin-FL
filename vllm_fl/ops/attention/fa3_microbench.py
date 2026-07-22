# Copyright (c) 2025 BAAI. All rights reserved.
"""Step-0 gate microbench for the PPU-native FlashAttention-3 adapter.

Purpose: BEFORE touching serving, verify on the actual PPU device that the
standalone ``flash_attn_interface`` package can serve vLLM's paged attention
at the shapes Qwen3.5-MoE uses, and pin down the exact ``q`` rank that
``flash_attn_with_kvcache`` expects.

Run:  python -m vllm_fl.ops.attention.fa3_microbench
"""

import math

import torch


NHEADS_Q = 16
NHEADS_KV = 2
HEAD_DIM = 256          # Qwen3.5-MoE full-attention head_dim
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _build_paged_cache(seqlens, block_size, generator):
    num_seqs = len(seqlens)
    max_blocks = max((s + block_size - 1) // block_size for s in seqlens)
    total_blocks = num_seqs * max_blocks + 1
    k_cache = torch.randn(total_blocks, block_size, NHEADS_KV, HEAD_DIM,
                          dtype=DTYPE, device=DEVICE, generator=generator)
    v_cache = torch.randn(total_blocks, block_size, NHEADS_KV, HEAD_DIM,
                          dtype=DTYPE, device=DEVICE, generator=generator)
    page_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=DEVICE)
    blk = 1
    ref_kv = []
    for i, s in enumerate(seqlens):
        nb = (s + block_size - 1) // block_size
        blocks = list(range(blk, blk + nb))
        blk += nb
        page_table[i, :nb] = torch.tensor(blocks, dtype=torch.int32, device=DEVICE)
        k_seq = torch.cat([k_cache[b] for b in blocks], dim=0)[:s]
        v_seq = torch.cat([v_cache[b] for b in blocks], dim=0)[:s]
        ref_kv.append((k_seq, v_seq))
    cache_seqlens = torch.tensor(seqlens, dtype=torch.int32, device=DEVICE)
    return k_cache, v_cache, page_table, cache_seqlens, ref_kv


def _ref_attention(q_per_seq, ref_kv, causal):
    outs = []
    scale = 1.0 / math.sqrt(HEAD_DIM)
    rep = NHEADS_Q // NHEADS_KV
    for q, (k, v) in zip(q_per_seq, ref_kv):
        lq = q.shape[0]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        qf = q.float().transpose(0, 1)
        kf = k.float().transpose(0, 1)
        vf = v.float().transpose(0, 1)
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
        lk = k.shape[0]
        if causal:
            row = torch.arange(lq, device=DEVICE).view(-1, 1) + (lk - lq)
            col = torch.arange(lk, device=DEVICE).view(1, -1)
            scores = scores.masked_fill(col > row, float("-inf"))
        probs = scores.softmax(dim=-1)
        o = torch.matmul(probs, vf).transpose(0, 1)
        outs.append(o)
    return torch.cat(outs, dim=0)


def _cuda_time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _run_case(name, seqlens, q_lens, block_size, causal, gen):
    import flash_attn_interface as fa3

    k_cache, v_cache, page_table, cache_seqlens, ref_kv = _build_paged_cache(
        seqlens, block_size, gen)
    total_q = sum(q_lens)
    q = torch.randn(total_q, NHEADS_Q, HEAD_DIM, dtype=DTYPE, device=DEVICE, generator=gen)
    cu_q = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=DEVICE)
    cu_q[1:] = torch.tensor(q_lens, dtype=torch.int32, device=DEVICE).cumsum(0)
    max_seqlen_q = max(q_lens)

    print(f"\n=== {name}: seqlens={seqlens} q_lens={q_lens} block_size={block_size} "
          f"causal={causal} | q.shape={tuple(q.shape)} kcache={tuple(k_cache.shape)} ===")

    def call():
        return fa3.flash_attn_with_kvcache(
            q=q, k_cache=k_cache, v_cache=v_cache,
            page_table=page_table, cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_q, max_seqlen_q=max_seqlen_q,
            causal=causal,
        )

    try:
        out = call()
        out = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  [FAIL] flash_attn_with_kvcache raised: {type(e).__name__}: {str(e)[:300]}")
        return False

    if not torch.isfinite(out).all():
        print("  [FAIL] output not finite")
        return False
    print(f"  [ok] out.shape={tuple(out.shape)} finite=True")

    q_per_seq = list(torch.split(q, q_lens, dim=0))
    ref = _ref_attention(q_per_seq, ref_kv, causal).to(out.dtype)
    if ref.shape != out.shape:
        print(f"  [warn] ref shape {tuple(ref.shape)} != out {tuple(out.shape)}; skip numeric")
    else:
        diff = (out.float() - ref.float()).abs()
        rel = diff.max().item() / (ref.float().abs().max().item() + 1e-6)
        print(f"  numeric vs SDPA: max_abs={diff.max().item():.4f} max_rel={rel:.4f} "
              f"-> {'PASS' if rel < 0.03 else 'CHECK'}")

    ms = _cuda_time(call)
    print(f"  FA3 latency: {ms*1000:.1f} us/iter")
    return True


def main():
    print("device:", torch.cuda.get_device_name(0))
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    ok = True
    ok &= _run_case("decode bs=8 ctx=4096 blk=256",
                    seqlens=[4096]*8, q_lens=[1]*8, block_size=256, causal=True, gen=gen)
    ok &= _run_case("prefill bs=2 len=512 blk=256",
                    seqlens=[512]*2, q_lens=[512]*2, block_size=256, causal=True, gen=gen)
    ok &= _run_case("decode bs=8 ctx=4096 blk=512",
                    seqlens=[4096]*8, q_lens=[1]*8, block_size=512, causal=True, gen=gen)
    print("\n### R1 probe: block sizes NOT multiples of 256 ###")
    for bs in (128, 1056):
        _run_case(f"R1 blk={bs}", seqlens=[1056]*4, q_lens=[1]*4,
                  block_size=bs, causal=True, gen=gen)
    print("\nGATE:", "PASS (256-multiple cases ok)" if ok else "FAIL")


if __name__ == "__main__":
    main()
