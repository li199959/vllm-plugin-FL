import pytest

from vllm_fl import optimizations
from vllm_fl.optimizations.ppu import minimax_m3

_SWITCHES = (
    "VLLM_FL_MOE",
    "VLLM_FL_M3_DENSE",
    "VLLM_FL_M3_GEMMA_NORM",
    "VLLM_FL_M3_FUSED_IMPL",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in _SWITCHES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_disabled_by_default(clean_env):
    assert not minimax_m3.enabled()
    assert optimizations.select_moe_experts() is None


def test_moe_selection_and_vendor_gate(clean_env):
    from vllm.platforms import current_platform

    from vllm_fl.optimizations.ppu.minimax_m3.moe import HybridM3Experts

    clean_env.setenv("VLLM_FL_MOE", "hybrid")
    assert minimax_m3.enabled()
    expected = HybridM3Experts if current_platform.vendor_name == "thead" else None
    assert optimizations.select_moe_experts() is expected
    clean_env.setattr(current_platform, "vendor_name", "nvidia", raising=False)
    assert optimizations.select_moe_experts() is None


def test_deepgemm_switch_not_claimed(clean_env):
    clean_env.setenv("VLLM_FL_MOE", "deepgemm")
    assert optimizations.select_moe_experts() is None
