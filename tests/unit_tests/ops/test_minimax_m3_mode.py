import sys
from types import SimpleNamespace

import pytest

from vllm_fl.ops import minimax_m3_ops as ops


@pytest.mark.parametrize("result", [True, False])
def test_probe_reference_does_not_reenter_probe(monkeypatch, result):
    monkeypatch.setattr(ops, "_TRITON_MODE", None)
    calls = []

    def probe():
        calls.append(1)
        assert ops.resolve_triton_mode() is False
        return result

    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.ops.minimax_m3_triton",
        SimpleNamespace(probe_triton_m3_fused=probe),
    )
    assert ops.resolve_triton_mode() is result
    assert ops.resolve_triton_mode() is result
    assert calls == [1]


def test_failed_probe_keeps_torch_fallback(monkeypatch):
    monkeypatch.setattr(ops, "_TRITON_MODE", None)

    def probe():
        raise RuntimeError("Unsupported runtime")

    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.ops.minimax_m3_triton",
        SimpleNamespace(probe_triton_m3_fused=probe),
    )
    assert ops.resolve_triton_mode() is False
