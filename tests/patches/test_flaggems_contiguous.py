import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("src,dst", [(64, 40), (16, 10), (24, 64)])
@torch.inference_mode()
def test_bicubic_aa_strided_input_matches_aten(src, dst):
    flag_gems = pytest.importorskip("flag_gems")
    from vllm_fl.patches.flaggems_contiguous import apply_flaggems_contiguous_patches

    assert apply_flaggems_contiguous_patches()
    torch.manual_seed(src)
    x = torch.randn(1, src, src, 96, device="cuda").permute(0, 3, 1, 2)  # NHWC view
    ref = F.interpolate(x.contiguous(), size=(dst, dst), mode="bicubic", antialias=True)
    lib = torch.library.Library("aten", "IMPL")
    flag_gems.only_enable(lib=lib, include=["_upsample_bicubic2d_aa"])
    try:
        out = F.interpolate(x, size=(dst, dst), mode="bicubic", antialias=True)
    finally:
        del lib
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_sum_empty_input_returns_zero():
    flag_gems = pytest.importorskip("flag_gems")
    from vllm_fl.patches.flaggems_contiguous import apply_flaggems_contiguous_patches

    assert apply_flaggems_contiguous_patches()
    lib = torch.library.Library("aten", "IMPL")
    flag_gems.only_enable(lib=lib, include=["sum"])
    try:
        # DeepSeek-OCR: torch.sum(patches) on an image with no crop tiles.
        empty = torch.empty(0, 3, 640, 640, device="cuda", dtype=torch.bfloat16)
        out = torch.sum(empty)
        assert out.item() == 0 and out.dtype == torch.bfloat16
        assert (
            torch.sum(torch.empty(0, dtype=torch.bool, device="cuda")).dtype
            == torch.int64
        )
        x = torch.randn(4096, device="cuda")
        torch.testing.assert_close(
            torch.sum(x), x.double().sum().float(), atol=1e-2, rtol=1e-4
        )
    finally:
        del lib


@pytest.mark.parametrize("batched", [False, True])
@torch.inference_mode()
def test_conv1d_unbatched_input_matches_aten(batched):
    flag_gems = pytest.importorskip("flag_gems")
    from vllm_fl.patches.flaggems_contiguous import apply_flaggems_contiguous_patches

    assert apply_flaggems_contiguous_patches()
    torch.manual_seed(0)
    # Whisper-large-v3 encoder: conv1 (128 -> 1280, k3 p1), conv2 (k3 s2 p1).
    x = torch.randn(128, 3000, device="cuda", dtype=torch.float16)
    if batched:
        x = x.unsqueeze(0)
    w1 = torch.randn(1280, 128, 3, device="cuda", dtype=torch.float16) * 0.05
    w2 = torch.randn(1280, 1280, 3, device="cuda", dtype=torch.float16) * 0.02
    b = torch.randn(1280, device="cuda", dtype=torch.float16)
    ref1 = F.conv1d(x, w1, b, padding=1)
    ref2 = F.conv1d(ref1, w2, b, stride=2, padding=1)
    lib = torch.library.Library("aten", "IMPL")
    flag_gems.only_enable(lib=lib, include=["conv1d"])
    try:
        out1 = F.conv1d(x, w1, b, padding=1)
        out2 = F.conv1d(ref1, w2, b, stride=2, padding=1)
    finally:
        del lib
    assert out1.shape == ref1.shape and out2.shape == ref2.shape
    torch.testing.assert_close(out1, ref1, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out2, ref2, atol=2e-2, rtol=2e-2)


def test_patch_is_idempotent():
    flag_gems = pytest.importorskip("flag_gems")
    from vllm_fl.patches.flaggems_contiguous import apply_flaggems_contiguous_patches

    apply_flaggems_contiguous_patches()
    apply_flaggems_contiguous_patches()
    for name in ("_upsample_bicubic2d_aa", "sum", "conv1d", "conv1d.padding"):
        entries = [e for e in flag_gems._FULL_CONFIG if e[0] == name]
        assert len(entries) == 1 and hasattr(entries[0][1], "wrapped")
        assert not hasattr(entries[0][1].wrapped, "wrapped")
