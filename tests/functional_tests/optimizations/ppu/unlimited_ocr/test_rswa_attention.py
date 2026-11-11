import pytest
import torch

from vllm_fl.models.unlimited_ocr.rswa_unified_attention import unified_attention

pytestmark = pytest.mark.gpu

H, D, BS = 10, 128, 16  # Unlimited-OCR text backbone: 10 heads x 128, block 16


def _paged(seq_lens):
    torch.manual_seed(0)
    num_blocks = len(seq_lens) * ((max(seq_lens) + BS - 1) // BS) + 8
    k_cache = torch.randn(num_blocks, BS, H, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, BS, H, D, device="cuda", dtype=torch.bfloat16)
    max_blocks = max((s + BS - 1) // BS for s in seq_lens)
    # Shuffle on CPU: other tests in the session may leave FlagGems registered for
    # CUDA aten ops, and the table must not depend on them.
    perm = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(0))
    table = perm[: len(seq_lens) * max_blocks].view(len(seq_lens), max_blocks).int()
    # vLLM pads each row to max_model_len / block_size; the kernel may read a few
    # entries past the last used block (masked tiles), so pad like the runner does.
    table = torch.cat([table, torch.zeros(len(seq_lens), 32, dtype=torch.int32)], 1)
    return k_cache, v_cache, table.cuda()


def _gather(cache, table_row, n):
    blocks = cache[table_row[: (n + BS - 1) // BS].long()]
    return blocks.reshape(-1, H, D)[:n].float()


def _reference(q, k_cache, v_cache, table, q_lens, seq_lens, prefix_lens, window):
    outs, start = [], 0
    for i, (ql, sl) in enumerate(zip(q_lens, seq_lens)):
        k = _gather(k_cache, table[i], sl)
        v = _gather(v_cache, table[i], sl)
        qi = q[start : start + ql].float()
        qpos = torch.arange(sl - ql, sl, device="cuda")[:, None]
        kpos = torch.arange(sl, device="cuda")[None, :]
        mask = kpos <= qpos
        if window is not None:
            mask &= (kpos < prefix_lens[i]) | ((qpos - kpos) < window)
        s = torch.einsum("qhd,khd->hqk", qi, k) * D**-0.5
        s = s.masked_fill(~mask[None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", s.softmax(-1), v))
        start += ql
    return torch.cat(outs)


def _run(q_lens, seq_lens, prefix_lens, window, use_3d):
    k_cache, v_cache, table = _paged(seq_lens)
    q = torch.randn(sum(q_lens), H, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    # Index tensors are built on CPU (see _paged).
    cu = torch.tensor([0] + list(q_lens)).cumsum(0).int().cuda()
    seqused = torch.tensor(seq_lens, dtype=torch.int32).cuda()
    prefix = torch.tensor(prefix_lens, dtype=torch.int32).cuda()
    extra = {}
    if use_3d:
        n, segs = len(seq_lens), 16
        extra = dict(
            seq_threshold_3D=n,
            num_par_softmax_segments=segs,
            softmax_segm_output=torch.empty(n, H, segs, D, device="cuda"),
            softmax_segm_max=torch.empty(n, H, segs, device="cuda"),
            softmax_segm_expsum=torch.empty(n, H, segs, device="cuda"),
        )
    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu,
        max_seqlen_q=max(q_lens),
        seqused_k=seqused,
        max_seqlen_k=max(seq_lens),
        softmax_scale=D**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        rswa_prefix_lens=prefix if window else None,
        rswa_window=window,
        **extra,
    )
    ref = _reference(q, k_cache, v_cache, table, q_lens, seq_lens, prefix_lens, window)
    return out.float(), ref


CASES = {
    # decode far past the window, decode inside the window, decode at the boundary
    "decode": (
        [1, 1, 1, 1],
        [3000, 900, 1000 + 128, 1000 + 129],
        [800, 850, 1000, 1000],
    ),
    # prefill (all queries in the prompt) and chunked prefill continuation
    "prefill": ([700, 300], [700, 1500], [700, 1500]),
    # mixed: decode + prefill in one batch
    "mixed": ([1, 400, 1], [2500, 400, 1300], [600, 400, 1000]),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("use_3d", [False, True])
@torch.inference_mode()
def test_rswa_matches_reference(case, use_3d):
    q_lens, seq_lens, prefix_lens = CASES[case]
    if use_3d and max(q_lens) > 1:
        pytest.skip("3D path is decode-only")
    out, ref = _run(q_lens, seq_lens, prefix_lens, 128, use_3d)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_rswa_differs_from_causal_past_window():
    # Guard against the mask silently not applying.
    q_lens, seq_lens, prefix_lens = [1], [3000], [800]
    out, ref_rswa = _run(q_lens, seq_lens, prefix_lens, 128, False)
    _, ref_causal = _run(q_lens, seq_lens, prefix_lens, None, False)
    torch.testing.assert_close(out, ref_rswa, atol=2e-2, rtol=2e-2)
    assert (ref_rswa - ref_causal).abs().max() > 0.05


@torch.inference_mode()
def test_no_rswa_is_plain_causal():
    q_lens, seq_lens, prefix_lens = CASES["mixed"]
    out, ref = _run(q_lens, seq_lens, prefix_lens, None, False)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
