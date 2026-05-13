# Copyright (c) 2025 BAAI. All rights reserved.

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm")

from vllm_fl.worker.memory_snapshot_profiler import (
    MemorySnapshotProfiler,
    get_memory_profiler_dir,
    get_memory_profiler_dump_on_exception,
    get_memory_profiler_max_entries,
    get_memory_profiler_profile_init,
    get_memory_profiler_settings,
    memory_profiler_enabled,
)


class FakeTorchDevice:
    def __init__(self):
        self.memory = SimpleNamespace(
            _record_memory_history=Mock(),
            _dump_snapshot=Mock(),
        )
        self.empty_cache = Mock()
        self.reset_peak_memory_stats = Mock()
        self.synchronize = Mock()


def test_memory_profiler_config_env(monkeypatch, tmp_path):
    out_dir = tmp_path / "snapshots"
    monkeypatch.setenv("VLLM_FL_MEMORY_PROFILER_DIR", str(out_dir))
    monkeypatch.setenv("VLLM_FL_MEMORY_PROFILER_MAX_ENTRIES", "7")
    monkeypatch.setenv("VLLM_FL_MEMORY_PROFILER_PROFILE_INIT", "0")
    monkeypatch.setenv("VLLM_FL_MEMORY_PROFILER_DUMP_ON_EXCEPTION", "false")

    settings = get_memory_profiler_settings()

    assert memory_profiler_enabled()
    assert settings.output_dir == os.path.abspath(str(out_dir))
    assert settings.max_entries == 7
    assert not settings.profile_init
    assert not settings.dump_on_exception
    assert get_memory_profiler_dir() == settings.output_dir
    assert get_memory_profiler_max_entries() == settings.max_entries
    assert get_memory_profiler_profile_init() == settings.profile_init
    assert get_memory_profiler_dump_on_exception() == settings.dump_on_exception


def test_memory_snapshot_profiler_start_stop(monkeypatch, tmp_path):
    fake_device = FakeTorchDevice()
    monkeypatch.setattr(
        "vllm_fl.worker.memory_snapshot_profiler.current_platform",
        SimpleNamespace(torch_device_fn=fake_device),
    )

    profiler = MemorySnapshotProfiler(
        output_dir=str(tmp_path),
        filename_prefix="test_memory",
        max_entries=13,
    )
    profiler.set_rank(2)

    assert profiler.start() is profiler
    assert profiler.is_recording
    fake_device.memory._record_memory_history.assert_called_once_with(
        enabled="all",
        stacks="all",
        max_entries=13,
    )

    snapshot_file = profiler.stop(suffix="manual")

    assert snapshot_file is not None
    assert os.path.basename(snapshot_file).startswith("test_memory_rank2_manual_")
    assert snapshot_file.endswith(".pickle")
    fake_device.synchronize.assert_called_once()
    fake_device.memory._dump_snapshot.assert_called_once_with(snapshot_file)
    assert fake_device.memory._record_memory_history.call_args_list[-1].kwargs == {
        "enabled": None
    }
    assert not profiler.is_recording


def test_memory_snapshot_profiler_requires_snapshot_api(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "vllm_fl.worker.memory_snapshot_profiler.current_platform",
        SimpleNamespace(torch_device_fn=SimpleNamespace(memory=SimpleNamespace())),
    )

    profiler = MemorySnapshotProfiler(output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="Memory snapshot profiling requires"):
        profiler.start()
