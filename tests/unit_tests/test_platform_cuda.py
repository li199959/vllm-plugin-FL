from types import SimpleNamespace

import pytest

from vllm_fl.platform import PlatformFL


@pytest.mark.parametrize(
    "vendor,device_type,expected",
    [
        ("nvidia", "cuda", True),
        ("thead", "cuda", True),
        ("hygon", "cuda", False),
        ("iluvatar", "cuda", False),
        ("metax", "cuda", False),
        ("unknown", "cuda", False),
        ("ascend", "npu", False),
        ("mthreads", "musa", False),
        ("thead", "cpu", False),
    ],
)
def test_cuda_paths_require_a_supported_vendor(vendor, device_type, expected):
    platform = SimpleNamespace(vendor_name=vendor, device_type=device_type)
    assert PlatformFL.is_cuda(platform) is expected
