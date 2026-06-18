# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tokens", [1, 8, 127])
@pytest.mark.parametrize("make_noncontiguous", [False, True])
def test_fused_qkvzba_split_reshape_contiguous(
    num_tokens: int,
    make_noncontiguous: bool,
) -> None:
    from vllm_fl.patches.gdn_fused_projection import (
        fused_qkvzba_split_reshape_contiguous,
    )

    num_heads_qk = 2
    num_heads_v = 8
    head_qk = 128
    head_v = 128
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v

    mixed_qkvz = torch.randn(
        num_tokens,
        qkv_dim + num_heads_v * head_v,
        device="cuda",
        dtype=torch.bfloat16,
    )
    mixed_ba = torch.randn(
        num_tokens,
        num_heads_v * 2,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if make_noncontiguous:
        mixed_qkvz = torch.stack((mixed_qkvz, mixed_qkvz), dim=-1)[..., 0]
        mixed_ba = torch.stack((mixed_ba, mixed_ba), dim=-1)[..., 0]
        assert not mixed_qkvz.is_contiguous()
        assert not mixed_ba.is_contiguous()

    actual_qkv, actual_z, actual_b, actual_a = fused_qkvzba_split_reshape_contiguous(
        mixed_qkvz,
        mixed_ba,
        num_heads_qk=num_heads_qk,
        num_heads_v=num_heads_v,
        head_qk=head_qk,
        head_v=head_v,
    )

    expected_qkv, expected_z = mixed_qkvz.split([qkv_dim, num_heads_v * head_v], dim=-1)
    expected_z = expected_z.reshape(num_tokens, num_heads_v, head_v)
    expected_b, expected_a = mixed_ba.chunk(2, dim=-1)

    torch.testing.assert_close(actual_qkv, expected_qkv)
    torch.testing.assert_close(actual_z, expected_z)
    torch.testing.assert_close(actual_b, expected_b)
    torch.testing.assert_close(actual_a, expected_a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_qkvzba_rejects_wrong_projection_width() -> None:
    from vllm_fl.patches.gdn_fused_projection import (
        fused_qkvzba_split_reshape_contiguous,
    )

    mixed_qkvz = torch.empty((1, 31), device="cuda")
    mixed_ba = torch.empty((1, 8), device="cuda")

    with pytest.raises(ValueError, match="QKVZ projection width"):
        fused_qkvzba_split_reshape_contiguous(
            mixed_qkvz,
            mixed_ba,
            num_heads_qk=2,
            num_heads_v=4,
            head_qk=4,
            head_v=2,
        )
