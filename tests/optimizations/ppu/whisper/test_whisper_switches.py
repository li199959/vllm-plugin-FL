import pytest

from vllm_fl.optimizations import ppu
from vllm_fl.optimizations.ppu import whisper

SWITCHES = (
    "VLLM_FL_WHISPER_COMPILE_ENCODER",
    "VLLM_FL_WHISPER_COMPILE_PREFILL",
    "VLLM_FL_WHISPER_STABLE",
    "VLLM_FL_WHISPER_CROSS_SLOTS",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_registered_in_ppu_models():
    assert whisper in ppu.model_packages()


@pytest.mark.parametrize(
    "name,value,expected",
    [
        (None, None, False),
        ("VLLM_FL_WHISPER_COMPILE_ENCODER", "0", False),
        ("VLLM_FL_WHISPER_COMPILE_ENCODER", "1", True),
        ("VLLM_FL_WHISPER_COMPILE_PREFILL", "on", True),
        ("VLLM_FL_WHISPER_STABLE", "1", True),
        ("VLLM_FL_WHISPER_CROSS_SLOTS", "1", True),
    ],
)
def test_enabled_switch(clean_env, name, value, expected):
    if name:
        clean_env.setenv(name, value)
    assert whisper.enabled() is expected


def test_install_patches_only_whisper_model(clean_env):
    from vllm.model_executor.models import whisper as vw

    from vllm_fl.optimizations.ppu.whisper import compile as wc

    clean_env.setattr(wc, "_installed", False)
    clean_env.setattr(
        vw.WhisperModel, "get_encoder_outputs", vw.WhisperModel.get_encoder_outputs
    )
    clean_env.setattr(vw.WhisperModel, "forward", vw.WhisperModel.forward)
    orig_encoder_forward = vw.WhisperEncoder.forward
    orig_forward = vw.WhisperModel.forward
    wc.install(encoder=True, prefill=True)
    assert vw.WhisperModel.get_encoder_outputs is wc._get_encoder_outputs
    assert vw.WhisperModel.forward is wc._model_forward
    assert vw.WhisperModel._fl_orig_forward is orig_forward
    assert (
        vw.WhisperEncoder.forward is orig_encoder_forward
    )  # shared by other audio models
    wc.install(encoder=True, prefill=True)  # idempotent
    assert vw.WhisperModel._fl_orig_forward is orig_forward
