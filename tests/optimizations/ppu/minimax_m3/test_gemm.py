"""Check the existing Gems GEMMs before enabling them for MiniMax-M3."""

import flag_gems
import pytest
import torch

from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("rows", [1, 64, 257])
@pytest.mark.parametrize(
    "k,n,dtype",
    [
        (6144, 1024, torch.bfloat16),
        (512, 6144, torch.bfloat16),
        (6144, 12504, torch.bfloat16),
        (6144, 128, torch.float32),
    ],
)
@torch.inference_mode()
def test_gems_projection_and_graph(rows, k, n, dtype):
    _disable_flaggems_runtime_autotune()
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(rows, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype) * 0.02
    bias = torch.randn(n, device="cuda", dtype=dtype) * 0.1
    reference = torch.mm(x.float(), weight.float().T)
    atol, rtol = (0.025, 0.016) if dtype == torch.bfloat16 else (0.001, 0.001)
    for op, expected in [
        (lambda: flag_gems.mm(x, weight.T), reference),
        (lambda: flag_gems.linear(x, weight), reference),
        (lambda: flag_gems.addmm(bias, x, weight.T), reference + bias.float()),
    ]:
        out = op()
        torch.testing.assert_close(out.float(), expected, rtol=rtol, atol=atol)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, rtol=rtol, atol=atol)
