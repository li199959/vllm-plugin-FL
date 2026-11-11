import sys

import pytest

from vllm_fl.optimizations import ppu
from vllm_fl.optimizations.ppu import qwen3_vl_embedding
from vllm_fl.optimizations.ppu.qwen3_vl_embedding import linear

SWITCHES = (
    "VLLM_FL_Q3VLEMB_STABLE",
    "VLLM_FL_Q3VLEMB_DENSE",
    "VLLM_FL_Q3VLEMB_GEMS_SMALL_M",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_registered_in_ppu_models():
    assert qwen3_vl_embedding in ppu.model_packages()


@pytest.mark.parametrize(
    "name,value,expected",
    [
        (None, None, False),
        ("VLLM_FL_Q3VLEMB_STABLE", "0", False),
        ("VLLM_FL_Q3VLEMB_STABLE", "1", True),
        ("VLLM_FL_Q3VLEMB_STABLE", "On", True),
        ("VLLM_FL_Q3VLEMB_DENSE", "hybrid", True),
        ("VLLM_FL_Q3VLEMB_DENSE", "tuned", False),
        (
            "VLLM_FL_Q3VLEMB_GEMS_SMALL_M",
            "512",
            False,
        ),  # only meaningful with DENSE=hybrid
    ],
)
def test_enabled_switch(clean_env, name, value, expected):
    if name:
        clean_env.setenv(name, value)
    assert qwen3_vl_embedding.enabled() is expected


@pytest.mark.parametrize("raw,expected", [(None, 0), ("", 0), ("512", 512)])
def test_small_m_threshold(clean_env, raw, expected):
    if raw is not None:
        clean_env.setenv("VLLM_FL_Q3VLEMB_GEMS_SMALL_M", raw)
    assert linear.small_m_from_env() == expected


@pytest.mark.parametrize("name", sorted(linear.SHAPES))
def test_tile_table_covers_every_m(name):
    for m in (1, 64, 65, 256, 512, 1 << 20):
        bm, bn, bk, warps, stages = linear.tile_for(name, m)
        assert bm in (64, 128) and bn in (64, 128, 256) and bk in (32, 64)


@pytest.mark.gpu
def test_register_installs_requested_parts(clean_env):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_fl.optimizations.ppu.common import shape_stable

    clean_env.setenv("VLLM_FL_Q3VLEMB_STABLE", "1")
    clean_env.setenv("VLLM_FL_Q3VLEMB_DENSE", "hybrid")
    clean_env.setenv("VLLM_FL_Q3VLEMB_GEMS_SMALL_M", "512")
    clean_env.setattr(qwen3_vl_embedding, "_registered", False)
    clean_env.setattr(linear, "_original_linear_apply", None)
    clean_env.setattr(UnquantizedLinearMethod, "apply", UnquantizedLinearMethod.apply)
    qwen3_vl_embedding.register()
    assert isinstance(
        sys.modules["flag_gems.ops.index"]._linearized_adjacent_index_func,
        shape_stable.ShapeStableIndexFunction,
    )
    assert isinstance(
        sys.modules["flag_gems.ops.copy"]._copy_kernel,
        shape_stable.ShapeStableCopyKernel,
    )
    assert UnquantizedLinearMethod.apply is linear._apply
    assert linear._small_m == 512
    qwen3_vl_embedding.register()  # idempotent
    assert UnquantizedLinearMethod.apply is linear._apply
