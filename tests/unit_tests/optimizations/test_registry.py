import types

import pytest

from vllm_fl import optimizations
from vllm_fl.optimizations import ppu


def _model(enabled, experts=None, vendor="thead"):
    calls = []
    return types.SimpleNamespace(
        VENDOR=vendor,
        enabled=lambda: enabled,
        register=lambda: calls.append(1),
        moe_experts=lambda: experts,
        calls=calls,
    )


@pytest.fixture
def platform(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "vendor_name", "thead", raising=False)
    return current_platform


def test_no_models_is_noop(monkeypatch, platform):
    monkeypatch.setattr(ppu, "model_packages", lambda: [])
    optimizations.register_model_optimizations()
    assert optimizations.select_moe_experts() is None


def test_only_enabled_models_register_and_select(monkeypatch, platform):
    on, off = _model(True, experts=object), _model(False, experts=int)
    monkeypatch.setattr(ppu, "model_packages", lambda: [off, on])
    optimizations.register_model_optimizations()
    assert on.calls == [1] and off.calls == []
    assert optimizations.select_moe_experts() is object


def test_other_vendor_is_skipped(monkeypatch, platform):
    m = _model(True, experts=object)
    monkeypatch.setattr(ppu, "model_packages", lambda: [m])
    monkeypatch.setattr(platform, "vendor_name", "nvidia", raising=False)
    optimizations.register_model_optimizations()
    assert m.calls == [] and optimizations.select_moe_experts() is None
