import pytest
import torch
from torch import nn

pytestmark = pytest.mark.gpu


class _Core(nn.Module):
    """Stands in for MMEncoderAttention; the patch replaces its forward."""

    num_heads = num_kv_heads = 4

    def __init__(self, d):
        super().__init__()
        self.scale = (d // 4) ** -0.5

    def forward(self, q, k, v):
        b, s, d = q.shape
        q, k, v = (t.view(b, s, 4, d // 4).transpose(1, 2) for t in (q, k, v))
        o = nn.functional.scaled_dot_product_attention(q, k, v)
        return o.transpose(1, 2).reshape(b, s, d)


class _SelfAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.attn = _Core(d)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.out(self.attn(q, k, v))


class _Layer(nn.Module):
    """WhisperEncoderLayer-shaped block without vLLM attention/parallel layers."""

    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = _SelfAttn(d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.fc2(nn.functional.gelu(self.fc1(self.ln2(x))))
        return torch.clamp(x, -6e4, 6e4)


class _Encoder(nn.Module):
    def __init__(self, mels=16, d=64, frames=40, layers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(frames // 2, d)
        self.layers = nn.ModuleList(_Layer(d) for _ in range(layers))
        self.layer_norm = nn.LayerNorm(d)

    def reference(self, feats):
        hs = []
        for f in feats:
            e = nn.functional.gelu(
                self.conv2(nn.functional.gelu(self.conv1(f)))
            ).transpose(-1, -2)
            hs.append(e + self.embed_positions.weight[: e.size(-2)])
        h = torch.stack(hs)
        for layer in self.layers:
            h = layer(h)
        return self.layer_norm(h)


@pytest.mark.parametrize("batches", [(3, 5, 1, 7)])
@torch.inference_mode()
def test_compiled_encoder_matches_eager_across_batch_sizes(batches):
    from vllm_fl.optimizations.ppu.whisper.compile import _encoder_forward

    torch.manual_seed(0)
    enc = _Encoder().cuda().eval()
    torch._dynamo.reset()
    for n in batches:
        feats = [torch.randn(16, 40, device="cuda") for _ in range(n)]
        out = _encoder_forward(enc, feats)
        ref = enc.reference(feats)
        assert out.shape == (n, 20, 64)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
    assert enc._fl_compiled_stack is not None
    assert enc.layers[0].self_attn.attn.forward.__name__ == "forward"  # patched closure
    assert "forward" in enc.layers[0].self_attn.attn.__dict__


@torch.inference_mode()
def test_encoder_graph_replay_matches_and_does_not_alias(monkeypatch):
    from vllm_fl.optimizations.ppu.whisper.compile import (
        _encoder_forward,
        _EncoderGraphs,
    )

    monkeypatch.setenv("VLLM_FL_WHISPER_ENCODER_GRAPH_MAX", "4")
    torch.manual_seed(1)
    enc = _Encoder().cuda().eval()
    torch._dynamo.reset()
    feats_a = [torch.randn(16, 40, device="cuda") for _ in range(3)]
    feats_b = [torch.randn(16, 40, device="cuda") for _ in range(3)]
    out_a = _encoder_forward(enc, feats_a)  # captures the batch-3 graph
    assert isinstance(enc._fl_encoder_runner, _EncoderGraphs)
    assert set(enc._fl_encoder_runner.graphs) == {3}
    keep = out_a.clone()
    out_b = _encoder_forward(enc, feats_b)  # replays it with new input
    torch.testing.assert_close(out_a, keep)  # earlier output is not the static buffer
    torch.testing.assert_close(out_b, enc.reference(feats_b), atol=2e-3, rtol=2e-3)
    big = [torch.randn(16, 40, device="cuda") for _ in range(6)]  # > max: no graph
    torch.testing.assert_close(
        _encoder_forward(enc, big), enc.reference(big), atol=2e-3, rtol=2e-3
    )
    assert set(enc._fl_encoder_runner.graphs) == {3}
