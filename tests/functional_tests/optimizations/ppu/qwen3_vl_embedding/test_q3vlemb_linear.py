# Copyright (c) 2026 BAAI. All rights reserved.
"""Hybrid Gems linear for Qwen3-VL-Embedding-8B: numerics, M split, CUDA graph replay."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def lin(monkeypatch):
    from vllm_fl.optimizations.ppu.qwen3_vl_embedding import linear

    monkeypatch.setattr(linear, "_small_m", 256)
    return linear


@pytest.mark.parametrize("name", ["qkv", "o", "gate_up", "down"])
@pytest.mark.parametrize("m", [1, 17, 64, 256])
def test_small_m_matches_cublas(lin, name, m):
    n, k = lin.SHAPES[name]
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    assert lin.projection(x, w) == name
    out = lin.gems_linear(x, w, None)
    ref = torch.nn.functional.linear(x, w)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.25, rtol=0.02)


def test_large_m_falls_back_to_cublas(lin):
    n, k = lin.SHAPES["o"]
    x = torch.randn(300, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    assert torch.equal(lin.gems_linear(x, w, None), torch.nn.functional.linear(x, w))


def test_other_shapes_and_dtypes_not_routed(lin):
    x = torch.randn(8, 1152, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4304, 1152, device="cuda", dtype=torch.bfloat16)
    assert lin.projection(x, w) is None
    n, k = lin.SHAPES["o"]
    assert (
        lin.projection(
            torch.randn(8, k, device="cuda"), torch.randn(n, k, device="cuda")
        )
        is None
    )


def test_disabled_when_small_m_zero(monkeypatch):
    from vllm_fl.optimizations.ppu.qwen3_vl_embedding import linear

    monkeypatch.setattr(linear, "_small_m", 0)
    n, k = linear.SHAPES["o"]
    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    assert (
        linear.projection(x, torch.randn(n, k, device="cuda", dtype=torch.bfloat16))
        is None
    )


def test_cuda_graph_replay_tracks_new_inputs(lin):
    n, k = lin.SHAPES["down"]
    x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    for _ in range(2):
        lin.gems_linear(x, w, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = lin.gems_linear(x, w, None)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float(), torch.nn.functional.linear(x, w).float(), atol=0.25, rtol=0.02
    )
