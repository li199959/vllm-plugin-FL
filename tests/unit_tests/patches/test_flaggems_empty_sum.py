import pytest
import torch

from vllm_fl.patches.flaggems_contiguous import _empty_sum


@pytest.mark.parametrize(
    "input_dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.complex64,
    ],
)
@pytest.mark.parametrize("output_dtype", [None, torch.int32, torch.float32])
def test_empty_sum_matches_aten(input_dtype, output_dtype):
    x = torch.empty((0, 3), dtype=input_dtype)
    expected = torch.sum(x, dtype=output_dtype)
    actual = _empty_sum(torch.sum, x, dtype=output_dtype)
    torch.testing.assert_close(actual, expected)


def test_nonempty_sum_keeps_original_implementation():
    x = torch.tensor([1, 2, 3], dtype=torch.int16)
    torch.testing.assert_close(_empty_sum(torch.sum, x), x.sum())
