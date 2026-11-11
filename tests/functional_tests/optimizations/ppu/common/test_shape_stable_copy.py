import sys

import pytest
import torch

from vllm_fl.ops.custom_ops import _disable_flaggems_runtime_autotune
from vllm_fl.optimizations.ppu.common.shape_stable import (
    _strided_copy_kernel,
    install_shape_stable_copy,
)

pytestmark = pytest.mark.gpu


def _compiled(jit_fn):
    while not hasattr(jit_fn, "device_caches"):
        jit_fn = jit_fn.fn
    return sum(len(cache[0]) for cache in jit_fn.device_caches.values())


@pytest.fixture(autouse=True)
def _setup():
    _disable_flaggems_runtime_autotune()
    install_shape_stable_copy()


def _gems_copy_(dst, src):
    import flag_gems  # noqa: F401

    return sys.modules["flag_gems.ops.copy"].copy_(dst, src)


@torch.inference_mode()
def test_mrope_slice_copies_compile_once():
    # Persistent [3, max_tokens + 1] buffer <-> contiguous [3, N] slices.
    buf = torch.arange(3 * 8193, device="cuda", dtype=torch.int64).view(3, 8193)
    sizes = (1, 7, 64, 129, 1000, 4096, 8192)
    for n in sizes:
        dst = torch.empty(3, n, device="cuda", dtype=torch.int64)
        _gems_copy_(dst, buf[:, :n])
        torch.testing.assert_close(dst, buf[:, :n], rtol=0, atol=0)
        back = torch.zeros_like(buf)
        _gems_copy_(back[:, :n], dst)
        torch.testing.assert_close(back[:, :n], buf[:, :n], rtol=0, atol=0)
        if n == sizes[0]:
            compiled = _compiled(_strided_copy_kernel)
    assert _compiled(_strided_copy_kernel) == compiled


@torch.inference_mode()
def test_permuted_logits_and_broadcast():
    torch.manual_seed(0)
    for b in (1, 13, 64):
        src = torch.randn(2, b, 1000, device="cuda").to(torch.bfloat16)
        dst = torch.empty(b, 2, 1000, device="cuda", dtype=torch.bfloat16)
        _gems_copy_(dst, src.permute(1, 0, 2))
        torch.testing.assert_close(dst, src.permute(1, 0, 2), rtol=0, atol=0)
    row = torch.arange(50, device="cuda", dtype=torch.int64)
    dst = torch.empty(3, 50, device="cuda", dtype=torch.int64)
    _gems_copy_(dst, row)
    torch.testing.assert_close(dst, row.expand(3, 50), rtol=0, atol=0)


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.int32, torch.int64),
        (torch.bool, torch.int32),
        (torch.float32, torch.bool),
    ],
)
@torch.inference_mode()
def test_dtype_cast_matches_torch(src_dtype, dst_dtype):
    torch.manual_seed(1)
    base = torch.randn(4, 33, 5, device="cuda") * 4
    src = base.to(src_dtype).transpose(0, 1)
    out = torch.empty(33, 4, 5, device="cuda", dtype=dst_dtype)
    _gems_copy_(out, src)
    expected = src.to(dst_dtype)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_contiguous_and_high_rank_fallback():
    x = torch.randn(6, 7, 8, device="cuda")
    out = torch.empty_like(x)
    _gems_copy_(out, x)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    y = torch.randn(2, 3, 4, 5, 6, device="cuda").transpose(1, 3)
    out = torch.empty(y.shape, device="cuda")
    _gems_copy_(out, y)
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@torch.inference_mode()
def test_graph_replay():
    buf = torch.arange(3 * 257, device="cuda", dtype=torch.int64).view(3, 257)
    dst = torch.zeros(3, 100, device="cuda", dtype=torch.int64)
    _gems_copy_(dst, buf[:, :100])  # compile outside capture
    dst.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _gems_copy_(dst, buf[:, :100])
    buf.add_(5)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, buf[:, :100], rtol=0, atol=0)
