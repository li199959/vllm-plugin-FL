"""Collect PPU kernel tests only on the vendor runtime they exercise."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _has_ppu():
    import torch

    from vllm.platforms import current_platform

    return (
        getattr(current_platform, "vendor_name", None) == "thead"
        and torch.cuda.is_available()
    )


def pytest_ignore_collect(collection_path, config):
    # Gate before importing tests: PPU DeepGEMM/FA3 APIs are not available
    # in the NVIDIA, NPU or CPU jobs that share the functional test scope.
    if collection_path.name.startswith("test_") and not _has_ppu():
        return True
    return None
