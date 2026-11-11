"""Numerical and graph checks against the native MiniMax implementations."""

import pytest
import torch

import vllm_fl._m3_fused_C  # noqa: F401
from vllm_fl.ops.minimax_m3_triton import (
    triton_fused_minimax_m3_qknorm_rope_kv_insert,
)
from vllm_fl.optimizations.ppu.minimax_m3.norm import (
    gemma_fused_add_rms_norm,
    gemma_rms_norm,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("rows", [1, 65, 257])
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_gemma_norm_native_and_graph(rows, with_residual):
    from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

    torch.manual_seed(31)
    width = 6144
    x0 = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    r0 = torch.randn_like(x0)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1
    x, residual = x0.clone(), r0.clone()
    native_x, native_r = x0.clone(), r0.clone()
    eps = 1e-6
    if with_residual:
        gemma_fused_add_rmsnorm(native_x, native_r, weight, eps)
    else:
        native_x = gemma_rmsnorm(native_x, weight, eps)

    def run():
        x.copy_(x0)
        residual.copy_(r0)
        if with_residual:
            gemma_fused_add_rms_norm(x, residual, weight, eps)
            return x
        return gemma_rms_norm(x, weight, eps)

    out = run()
    torch.testing.assert_close(out, native_x, rtol=0.016, atol=0.016)
    if with_residual:
        torch.testing.assert_close(residual, native_r, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, native_x, rtol=0.016, atol=0.016)
    if with_residual:
        torch.testing.assert_close(residual, native_r, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 65, 257])
@pytest.mark.parametrize("sparse,insert", [(False, False), (True, False), (True, True)])
@torch.inference_mode()
def test_m3_native_and_graph(rows, sparse, insert):
    torch.manual_seed(37)
    nq, nkv, niq, hd, rot = 4, 1, int(sparse), 128, 64
    row_heads = nq + 2 * nkv + (niq + 1 if sparse else 0)
    qkv0 = torch.randn(rows, row_heads * hd, device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn(hd, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(4)
    ]
    freqs = torch.outer(
        torch.arange(1024, device="cuda", dtype=torch.float32),
        1.0 / (5000000.0 ** (torch.arange(0, rot, 2, device="cuda") / rot)),
    )
    cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) + 7
    slots = torch.arange(rows, device="cuda", dtype=torch.int64) + 3
    index_slots = slots + 11
    slots[::7] = -1
    index_slots[::11] = -1
    block_size, num_blocks = 128, 4

    def buffers():
        return [
            qkv0.clone(),
            torch.full(
                (num_blocks, 2, block_size, nkv, hd),
                -7,
                device="cuda",
                dtype=torch.bfloat16,
            )
            if insert
            else None,
            torch.full(
                (num_blocks, block_size, hd), -7, device="cuda", dtype=torch.bfloat16
            )
            if insert
            else None,
            torch.empty(rows, nq * hd, device="cuda", dtype=torch.bfloat16)
            if insert
            else None,
            torch.empty(rows, niq * hd, device="cuda", dtype=torch.bfloat16)
            if insert
            else None,
        ]

    def invoke(fn, bufs):
        qkv, kv, idx, q_out, iq_out = bufs
        qkv.copy_(qkv0)
        fn(
            qkv,
            weights[0],
            weights[1],
            cos_sin,
            positions,
            nq,
            nkv,
            rot,
            1e-6,
            weights[2] if sparse else None,
            weights[3] if sparse else None,
            niq,
            slots if insert else None,
            index_slots if insert else None,
            kv,
            idx,
            block_size if insert else 0,
            q_out,
            iq_out,
            "auto",
        )

    reference, actual = buffers(), buffers()
    invoke(torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert, reference)
    invoke(triton_fused_minimax_m3_qknorm_rope_kv_insert, actual)

    def check():
        for a, b in zip(actual, reference):
            if a is not None:
                torch.testing.assert_close(a, b, rtol=0.016, atol=0.016)

    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            invoke(triton_fused_minimax_m3_qknorm_rope_kv_insert, actual)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke(triton_fused_minimax_m3_qknorm_rope_kv_insert, actual)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    check()
