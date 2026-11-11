import sys

import pytest

from vllm_fl.optimizations import ppu
from vllm_fl.optimizations.ppu import qwen3_vl_moe

SWITCHES = ("VLLM_FL_QWEN3VL_STABLE", "VLLM_FL_QWEN3VL_ATTN", "VLLM_FL_QWEN3VL_MOE")


@pytest.fixture
def clean_env(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_registered_in_ppu_models():
    assert qwen3_vl_moe in ppu.model_packages()


@pytest.mark.parametrize(
    "name,value,expected",
    [
        (None, None, False),
        ("VLLM_FL_QWEN3VL_STABLE", "0", False),
        ("VLLM_FL_QWEN3VL_STABLE", "1", True),
        ("VLLM_FL_QWEN3VL_STABLE", "ON", True),
        ("VLLM_FL_QWEN3VL_ATTN", "tuned", True),
        ("VLLM_FL_QWEN3VL_ATTN", "native", False),
        ("VLLM_FL_QWEN3VL_MOE", "Tuned", True),
    ],
)
def test_enabled_switch(clean_env, name, value, expected):
    if name:
        clean_env.setenv(name, value)
    assert qwen3_vl_moe.enabled() is expected


@pytest.mark.gpu
def test_register_installs_requested_parts(clean_env):
    from vllm_fl.optimizations.ppu.common import shape_stable

    clean_env.setenv("VLLM_FL_QWEN3VL_STABLE", "1")
    clean_env.setattr(qwen3_vl_moe, "_registered", False)
    qwen3_vl_moe.register()
    assert isinstance(
        sys.modules["flag_gems.ops.copy"]._copy_kernel,
        shape_stable.ShapeStableCopyKernel,
    )
    assert isinstance(
        sys.modules["flag_gems.ops.index"]._linearized_adjacent_index_func,
        shape_stable.ShapeStableIndexFunction,
    )
    qwen3_vl_moe.register()  # idempotent
