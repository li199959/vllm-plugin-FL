import types

import pytest
import torch

from vllm_fl.optimizations.ppu.bge_m3 import attention

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _fa3():
    pytest.importorskip("flash_attn_3._C")


@pytest.mark.parametrize("lens", [[1026] * 8, [2050, 17, 4098], [8192], [1, 513, 300]])
@torch.inference_mode()
def test_fa3_matches_triton_context_attention(lens):
    from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

    torch.manual_seed(len(lens))
    h, d = 16, 64
    t = sum(lens)
    q, k, v = (torch.randn(t, h, d, device="cuda", dtype=torch.float16) for _ in range(3))
    cu = torch.tensor([0] + lens, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    ref = torch.empty_like(q)
    context_attention_fwd(q=q, k=k, v=v, o=ref, b_start_loc=cu[:-1],
                          b_seq_len=cu[1:] - cu[:-1], max_input_len=max(lens),
                          is_causal=False, softmax_scale=d**-0.5)
    out = torch.empty_like(q)
    attention.fa3_encoder_attention(q, k, v, out, cu, max(lens), d**-0.5)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


def test_eligibility(monkeypatch):
    monkeypatch.setattr(attention, "_min_tokens", 2048)
    q = torch.empty(2048, 16, 64, dtype=torch.float16)
    full = types.SimpleNamespace(sliding_window=(-1, -1))
    window = types.SimpleNamespace(sliding_window=(127, 127))
    assert attention._fa3_eligible(full, q, q)
    assert not attention._fa3_eligible(full, q[:2047], q[:2047])  # small batch keeps Triton
    assert not attention._fa3_eligible(window, q, q)
    assert not attention._fa3_eligible(full, q.float(), q.float())


@pytest.mark.parametrize("raw,expected", [(None, 2048), ("", 2048), ("0", 0), ("4096", 4096)])
def test_min_tokens_env(monkeypatch, raw, expected):
    monkeypatch.delenv("VLLM_FL_BGEM3_FA3_MIN_TOKENS", raising=False)
    if raw is not None:
        monkeypatch.setenv("VLLM_FL_BGEM3_FA3_MIN_TOKENS", raw)
    assert attention.min_tokens_from_env() == expected


@torch.inference_mode()
def test_install_routes_encoder_attention(monkeypatch):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    monkeypatch.setattr(attention, "_installed", False)
    monkeypatch.setenv("VLLM_FL_BGEM3_FA3_MIN_TOKENS", "0")
    monkeypatch.setattr(TritonAttentionImpl, "_forward_encoder_attention",
                        TritonAttentionImpl._forward_encoder_attention)
    attention.install_fa3_encoder_attention()
    patched = TritonAttentionImpl._forward_encoder_attention
    assert hasattr(patched, "triton")
    h, d, n = 16, 64, 300
    q, k, v = (torch.randn(n, h, d, device="cuda", dtype=torch.float16) for _ in range(3))
    meta = types.SimpleNamespace(query_start_loc=torch.tensor([0, 100, 300], device="cuda",
                                                               dtype=torch.int32),
                                 seq_lens=torch.tensor([100, 200], device="cuda",
                                                       dtype=torch.int32),
                                 max_query_len=200)
    impl = types.SimpleNamespace(sliding_window=(-1, -1), scale=d**-0.5,
                                 kv_cache_dtype="auto")
    out, ref = torch.empty_like(q), torch.empty_like(q)
    patched(impl, q, k, v, out, meta, None)
    patched.triton(impl, q, k, v, ref, meta, None)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
