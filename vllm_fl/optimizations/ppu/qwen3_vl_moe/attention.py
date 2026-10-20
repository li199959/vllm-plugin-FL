"""Launch params for vLLM's Triton unified attention on PPU (Qwen3-VL, TP2).

unified_attention leaves num_warps/num_stages to Triton's defaults (4/3). On
PPU, decode with 16 query / 2 KV heads per rank and 4-5k contexts is 1.6x
faster at 2 warps / 2 stages (tools/ppu/qwen3_vl_moe/sweep_attention.py).
Only the launch config changes; the kernel and its tiles are vLLM's.
"""
import torch

_installed = False

# grid rank -> launch kwargs. 3 = split-softmax decode (num_seqs <= threshold),
# 2 = 2D kernel (prefill/mixed batches or larger decode batches).
LAUNCH = {
    3: {"num_warps": 2, "num_stages": 2},
}


class _TunedLaunch:
    """Stands in for ``kernel_unified_attention``; ``[grid](...)`` adds kwargs."""

    def __init__(self, kernel, table):
        self.kernel, self.table = kernel, table

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        extra = self.table.get(len(grid)) if isinstance(grid, tuple) else None
        if not extra:
            return launch

        def run(*args, **kwargs):
            for key, value in extra.items():
                kwargs.setdefault(key, value)
            return launch(*args, **kwargs)

        return run


def install_tuned_attention(table=None):
    global _installed
    if _installed:
        return
    import vllm.v1.attention.ops.triton_unified_attention as ua

    ua.kernel_unified_attention = _TunedLaunch(ua.kernel_unified_attention, table or LAUNCH)
    _installed = True


# --- hybrid: PPU FA3 for steps that contain prefill -------------------------
# Triton's 2D (prefill/mixed) kernel is 3.4-4.2x slower than the PPU FA3 wheel
# on 16k chunked-prefill steps (tools/ppu/qwen3_vl_moe/bench_prefill_attention.py),
# which stalls every decoding request for ~2.5 s per step at 16k/1k. Pure
# decode steps (max_seqlen_q == 1, FULL CUDA graphs) stay on Triton. Mixed
# steps run in PIECEWISE graphs, where attention executes eagerly.

_hybrid_installed = False


def _fa3_eligible(kw, decode=False):
    # decode=True also routes pure-decode steps (FA3 is 1.3-2.2x the tuned
    # Triton 3D kernel at B=32-128, tools/ppu/qwen3_vl_moe/bench_decode_attention.py).
    return (
        (decode or kw.get("max_seqlen_q", 1) > 1)
        and kw.get("causal") is True
        and kw.get("window_size") in (None, (-1, -1), [-1, -1])
        and not kw.get("softcap")
        and kw.get("alibi_slopes") is None
        and kw.get("sinks") is None
        and kw.get("qq_bias") is None
        and kw.get("output_scale") is None
        and kw.get("mm_prefix_range") is None
        and kw.get("q_descale") is None
        and kw.get("k_scale_cache") is None
        and kw.get("chunk_lookback", -1) == -1
        and kw["q"].dtype in (torch.bfloat16, torch.float16)
        and kw["k"].dtype == kw["q"].dtype
    )


def fa3_paged_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k,
                        block_table, softmax_scale, num_splits=0):
    """Same call as vllm_fl's thead FA3 backend (PPU wants max_seqlen_k=1 when paged).

    num_splits=0 lets FA3 split KV (adds FlashAttnFwdCombine kernels); 1 disables it.
    """
    torch.ops.flash_attn_3.fwd(
        q, k, v, None, None, None, out, cu_seqlens_q, None, None, None, seqused_k,
        max_seqlen_q, 1, block_table, None, None, None, None, None, None, None, None,
        softmax_scale, True, -1, -1, 0, 0.0, True, None, num_splits, None, 0, None)
    return out


def _hybrid(triton_fn, decode=False):
    def unified_attention(**kw):
        if _fa3_eligible(kw, decode):
            # Steps with prefill: no KV split. Same speed on 16k mixed steps
            # (11.35 vs 11.34 ms/layer) and no FlashAttnFwdCombine<5>/<6> variants;
            # pure decode keeps FA3's own split choice (fixed splits are 9-15% slower).
            splits = 1 if kw["max_seqlen_q"] > 1 else 0
            return fa3_paged_attention(
                kw["q"], kw["k"], kw["v"], kw["out"], kw["cu_seqlens_q"],
                kw["max_seqlen_q"], kw["seqused_k"], kw["block_table"], kw["softmax_scale"],
                num_splits=splits)
        return triton_fn(**kw)

    unified_attention.triton = triton_fn
    return unified_attention


def install_hybrid_attention(decode=False):
    """decode=False: only steps with prefill go to FA3; True: every eligible step."""
    global _hybrid_installed
    if _hybrid_installed:
        return
    import flash_attn_3._C  # noqa: F401  (registers torch.ops.flash_attn_3)
    import vllm.v1.attention.backends.triton_attn as backend

    backend.unified_attention = _hybrid(backend.unified_attention, decode)
    _hybrid_installed = True
