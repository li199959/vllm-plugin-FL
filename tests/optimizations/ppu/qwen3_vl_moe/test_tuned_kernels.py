import sys

import pytest
import torch

from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune
from vllm_fl.optimizations.ppu.qwen3_vl_moe import attention, moe

pytestmark = pytest.mark.gpu

HQ, HKV, D, BS = 16, 2, 128, 16


@pytest.fixture(autouse=True)
def _setup():
    _disable_flaggems_runtime_autotune()


def _paged_decode(b, ctx=(300, 900), seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(ctx[0], ctx[1], (b,), generator=g, dtype=torch.int32)
    per = (lens + BS - 1) // BS
    nblocks = int(per.sum()) + 2
    k = (torch.randn(nblocks, BS, HKV, D, device="cuda") * 0.5).bfloat16()
    v = (torch.randn(nblocks, BS, HKV, D, device="cuda") * 0.5).bfloat16()
    table = torch.zeros(b, int(per.max()), dtype=torch.int32)
    perm = torch.randperm(nblocks - 2, generator=g, dtype=torch.int32)
    off = 0
    for i, n in enumerate(per.tolist()):
        table[i, :n] = perm[off:off + n]
        off += n
    q = (torch.randn(b, HQ, D, device="cuda") * 0.5).bfloat16()
    return q, k, v, table.cuda(), lens.cuda()


def _attention_call(q, k, v, table, lens, threshold=64, segments=16):
    """Allocate everything up front so the returned call is graph-capturable."""
    import vllm.v1.attention.ops.triton_unified_attention as ua

    b = q.shape[0]
    out = torch.empty_like(q)
    bufs = dict(
        seq_threshold_3D=threshold, num_par_softmax_segments=segments,
        softmax_segm_output=torch.empty(threshold, HQ, segments, D, device="cuda"),
        softmax_segm_max=torch.empty(threshold, HQ, segments, device="cuda"),
        softmax_segm_expsum=torch.empty(threshold, HQ, segments, device="cuda"),
    )
    cu_q = torch.arange(b + 1, dtype=torch.int32, device="cuda")
    max_len = int(lens.max())

    def call():
        ua.unified_attention(q, k, v, out, cu_q, 1, lens, max_len, D ** -0.5, True,
                             (-1, -1), table, 0, None, None, None, **bufs)
        return out

    return call


def _attend(q, k, v, table, lens):
    return _attention_call(q, k, v, table, lens)()


def _sdpa_reference(q, k, v, table, lens):
    outs = []
    for i in range(q.shape[0]):
        n = int(lens[i])
        blocks = table[i, : (n + BS - 1) // BS].long()
        ki = k[blocks].reshape(-1, HKV, D)[:n].float()
        vi = v[blocks].reshape(-1, HKV, D)[:n].float()
        qi = q[i].float().view(HKV, HQ // HKV, D)
        s = torch.einsum("hgd,nhd->hgn", qi, ki) * D ** -0.5
        outs.append(torch.einsum("hgn,nhd->hgd", s.softmax(-1), vi).reshape(HQ, D))
    return torch.stack(outs)


@torch.inference_mode()
def test_tuned_attention_matches_reference_both_paths(monkeypatch):
    import vllm.v1.attention.ops.triton_unified_attention as ua

    seen = []
    kernel = ua.kernel_unified_attention

    class Spy:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def run(*a, **kw):
                seen.append((len(grid), kw.get("num_warps"), kw.get("num_stages")))
                return launch(*a, **kw)
            return run

    monkeypatch.setattr(ua, "kernel_unified_attention",
                        attention._TunedLaunch(Spy(), attention.LAUNCH))
    for b in (5, 64, 80):  # 3D, 3D at the threshold, 2D above it
        q, k, v, table, lens = _paged_decode(b, seed=b)
        out = _attend(q, k, v, table, lens)
        torch.testing.assert_close(out.float(), _sdpa_reference(q, k, v, table, lens),
                                   atol=2e-2, rtol=2e-2)
    assert seen == [(3, 2, 2), (3, 2, 2), (2, None, None)]


@torch.inference_mode()
def test_tuned_attention_graph_replay(monkeypatch):
    import vllm.v1.attention.ops.triton_unified_attention as ua

    monkeypatch.setattr(ua, "kernel_unified_attention",
                        attention._TunedLaunch(ua.kernel_unified_attention, attention.LAUNCH))
    q, k, v, table, lens = _paged_decode(32, seed=7)
    call = _attention_call(q, k, v, table, lens)
    call()  # compile outside capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = call()
    q.mul_(0.5)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), _sdpa_reference(q, k, v, table, lens),
                               atol=2e-2, rtol=2e-2)


def _mixed_batch(q_len=300, prefix=200, decodes=5, ctx=400, seed=11):
    g = torch.Generator().manual_seed(seed)
    q_lens = [q_len] + [1] * decodes
    k_lens = [prefix + q_len] + [ctx + i for i in range(decodes)]
    per = [(n + BS - 1) // BS for n in k_lens]
    nblocks = sum(per) + 2
    k = (torch.randn(nblocks, BS, HKV, D, device="cuda") * 0.5).bfloat16()
    v = (torch.randn(nblocks, BS, HKV, D, device="cuda") * 0.5).bfloat16()
    table = torch.zeros(len(q_lens), max(per), dtype=torch.int32)
    perm = torch.randperm(nblocks - 2, generator=g, dtype=torch.int32)
    off = 0
    for i, n in enumerate(per):
        table[i, :n] = perm[off:off + n]
        off += n
    q = (torch.randn(sum(q_lens), HQ, D, device="cuda") * 0.5).bfloat16()
    cu_q = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32,
                        device="cuda")
    return q, k, v, table.cuda(), torch.tensor(k_lens, dtype=torch.int32, device="cuda"), \
        cu_q, max(q_lens), max(k_lens)


def _backend_kwargs(q, k, v, out, cu_q, max_q, lens, max_k, table):
    # Mirrors TritonAttentionImpl.forward's keyword call (no 3D buffers needed
    # for prefill steps).
    return dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, max_seqlen_q=max_q,
                seqused_k=lens, max_seqlen_k=max_k, softmax_scale=D ** -0.5, causal=True,
                alibi_slopes=None, use_alibi_sqrt=False, window_size=(-1, -1),
                block_table=table, softcap=0, q_descale=None, k_descale=None,
                v_descale=None, seq_threshold_3D=None, num_par_softmax_segments=None,
                softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None,
                sinks=None, output_scale=None, mm_prefix_range=None,
                k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td=False)


@torch.inference_mode()
def test_hybrid_prefill_uses_fa3_and_matches_triton():
    import flash_attn_3._C  # noqa: F401
    import vllm.v1.attention.ops.triton_unified_attention as ua

    calls = []

    def triton_spy(**kw):
        calls.append("triton")
        return ua.unified_attention(**kw)

    hybrid = attention._hybrid(triton_spy)
    q, k, v, table, lens, cu_q, max_q, max_k = _mixed_batch()
    ref, out = torch.empty_like(q), torch.empty_like(q)
    ua.unified_attention(**_backend_kwargs(q, k, v, ref, cu_q, max_q, lens, max_k, table))
    hybrid(**_backend_kwargs(q, k, v, out, cu_q, max_q, lens, max_k, table))
    assert calls == []  # went to FA3
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    # Ineligible (sliding window) and pure decode both stay on Triton.
    kw = _backend_kwargs(q, k, v, out, cu_q, max_q, lens, max_k, table)
    kw["window_size"] = (127, 0)
    hybrid(**kw)
    qd, kd, vd, td, ld = _paged_decode(4, seed=3)
    cu_d = torch.arange(5, dtype=torch.int32, device="cuda")
    hybrid(**_backend_kwargs(qd, kd, vd, torch.empty_like(qd), cu_d, 1, ld,
                             int(ld.max()), td))
    assert calls == ["triton", "triton"]


@torch.inference_mode()
def test_fa3_decode_matches_reference_and_graph_replay():
    # VLLM_FL_QWEN3VL_ATTN=fa3 sends pure decode (FULL CUDA graphs) to FA3 too.
    import flash_attn_3._C  # noqa: F401

    calls = []
    hybrid = attention._hybrid(lambda **kw: calls.append(1), decode=True)
    q, k, v, table, lens = _paged_decode(48, seed=5)
    cu = torch.arange(49, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    kw = _backend_kwargs(q, k, v, out, cu, 1, lens, int(lens.max()), table)
    hybrid(**kw)  # warm up outside capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        hybrid(**kw)
    q.mul_(0.5)
    g.replay()
    torch.cuda.synchronize()
    assert calls == []
    torch.testing.assert_close(out.float(), _sdpa_reference(q, k, v, table, lens),
                               atol=2e-2, rtol=2e-2)


def test_fa3_split_choice(monkeypatch):
    # Prefill/mixed steps run unsplit (no FlashAttnFwdCombine); decode keeps auto.
    seen = []
    monkeypatch.setattr(attention, "fa3_paged_attention",
                        lambda *a, num_splits: seen.append(num_splits))
    hybrid = attention._hybrid(lambda **kw: None, decode=True)
    q = torch.empty(4, HQ, D, dtype=torch.bfloat16)
    for max_q in (4, 1):
        hybrid(q=q, k=q, v=q, out=q, cu_seqlens_q=None, max_seqlen_q=max_q, seqused_k=None,
               block_table=None, softmax_scale=1.0, causal=True, window_size=(-1, -1),
               softcap=0)
    assert seen == [1, 0]


def test_fa3_eligibility_rules():
    q = torch.empty(4, HQ, D, dtype=torch.bfloat16)
    base = dict(q=q, k=q, max_seqlen_q=4, causal=True, window_size=(-1, -1), softcap=0)
    assert attention._fa3_eligible(base)
    for change in ({"max_seqlen_q": 1}, {"causal": False}, {"softcap": 30.0},
                   {"sinks": object()}, {"mm_prefix_range": object()},
                   {"q_descale": object()}, {"chunk_lookback": 2},
                   {"q": q.float(), "k": q.float()}):
        assert not attention._fa3_eligible({**base, **change}), change


def _moe_inputs(m, seed):
    torch.manual_seed(seed)
    e, h, i = moe.W1_SHAPE[0], moe.W1_SHAPE[2], moe.W2_SHAPE[2]
    w1 = (torch.randn(*moe.W1_SHAPE, device="cuda") * 0.02).bfloat16()
    w2 = (torch.randn(*moe.W2_SHAPE, device="cuda") * 0.02).bfloat16()
    x = (torch.randn(m, h, device="cuda") * 0.5).bfloat16()
    w, ids = torch.topk(torch.randn(m, e, device="cuda").softmax(-1), 8, dim=-1)
    return x, w1, w2, (w / w.sum(-1, keepdim=True)).float(), ids.to(torch.int32), i


def _moe_reference(x, w1, w2, tw, ids, inter):
    # FP32 math with the BF16 rounding of the activated intermediate that the
    # kernel also applies; grouped by expert so large M stays fast.
    out = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    flat = ids.reshape(-1).long()
    tokens = torch.arange(x.shape[0], device="cuda").repeat_interleave(ids.shape[1])
    weights = tw.reshape(-1)
    for e in flat.unique().tolist():
        sel = (flat == e).nonzero().squeeze(1)
        t = tokens[sel]
        gu = x[t].float() @ w1[e].float().t()
        act = (torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:]).bfloat16().float()
        out.index_add_(0, t, weights[sel, None] * (act @ w2[e].float().t()))
    return out


@torch.inference_mode()
def test_tuned_moe_lookup_scope(monkeypatch):
    import flag_gems  # noqa: F401

    fm = sys.modules["flag_gems.fused.fused_moe"]
    monkeypatch.setattr(moe, "_installed", False)
    monkeypatch.setattr(fm, "try_get_optimal_moe_config", fm.try_get_optimal_moe_config)
    gems = fm.try_get_optimal_moe_config
    moe.install_tuned_moe_tiles()
    tuned = fm.try_get_optimal_moe_config
    g1, g2 = moe.lookup_tiles(64)
    assert tuned(moe.W1_SHAPE, moe.W2_SHAPE, 8, "bf16", 64, 128) == g1
    assert tuned(moe.W1_SHAPE, moe.W2_SHAPE, 8, "bf16", 64, 128, gemm_stage="gemm2",
                 return_is_embedded=True) == (g2, False)
    assert tuned(moe.W1_SHAPE, moe.W2_SHAPE, 8, "bf16", 8192, 128) == moe.lookup_tiles(8192)[0]
    for args in [(moe.W1_SHAPE, moe.W2_SHAPE, 8, "bf16", 9000, 128),     # above the table
                 (moe.W1_SHAPE, moe.W2_SHAPE, 8, "bf16", 8, 128),        # below the table
                 (moe.W1_SHAPE, moe.W2_SHAPE, 4, "bf16", 64, 128),       # other top-k
                 ((128, 1536, 2048), (128, 2048, 768), 8, "bf16", 64, 128)]:  # TP1 shape
        assert tuned(*args) == gems(*args)


@pytest.mark.parametrize("m", [40, 64, 128, 129])
@torch.inference_mode()
def test_tuned_moe_matches_reference(monkeypatch, m):
    import flag_gems  # noqa: F401

    fm = sys.modules["flag_gems.fused.fused_moe"]
    monkeypatch.setattr(moe, "_installed", False)
    monkeypatch.setattr(fm, "try_get_optimal_moe_config", fm.try_get_optimal_moe_config)
    x, w1, w2, tw, ids, inter = _moe_inputs(m, seed=m)
    base = fm.fused_experts_impl(x, w1, w2, tw, ids)
    moe.install_tuned_moe_tiles()
    out = fm.fused_experts_impl(x, w1, w2, tw, ids)
    torch.testing.assert_close(out, base, rtol=0, atol=0)
    torch.testing.assert_close(out.float(), _moe_reference(x, w1, w2, tw, ids, inter),
                               atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("m", [300, 1000, 4096, 8192])
@torch.inference_mode()
def test_tuned_moe_prefill_sizes_match_reference(monkeypatch, m):
    # Gems' default at these sizes uses its GEMM fast path / direct-sum
    # reduction, so outputs are compared to the FP32 reference, not bitwise.
    import flag_gems  # noqa: F401

    fm = sys.modules["flag_gems.fused.fused_moe"]
    monkeypatch.setattr(moe, "_installed", False)
    monkeypatch.setattr(fm, "try_get_optimal_moe_config", fm.try_get_optimal_moe_config)
    x, w1, w2, tw, ids, inter = _moe_inputs(m, seed=m)
    ref = _moe_reference(x, w1, w2, tw, ids, inter)
    base = fm.fused_experts_impl(x, w1, w2, tw, ids)
    moe.install_tuned_moe_tiles()
    out = fm.fused_experts_impl(x, w1, w2, tw, ids)
    err_tuned = (out.float() - ref).abs().max().item()
    err_base = (base.float() - ref).abs().max().item()
    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-2)
    assert err_tuned <= max(2 * err_base, 1e-3), (err_tuned, err_base)
