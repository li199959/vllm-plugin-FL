# Copyright (c) 2025 BAAI. All rights reserved.
#
# Adapted from https://github.com/vllm-project/vllm/pull/30580
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@dataclass(frozen=True)
class MemoryProfilerSettings:
    output_dir: str = ""
    max_entries: int = 100000
    profile_init: bool = True
    dump_on_exception: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.output_dir)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _get_env_path(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    if not value:
        return ""
    return os.path.abspath(os.path.expanduser(str(value)))


def get_memory_profiler_settings() -> MemoryProfilerSettings:
    return MemoryProfilerSettings(
        output_dir=_get_env_path("VLLM_FL_MEMORY_PROFILER_DIR"),
        max_entries=_get_env_int("VLLM_FL_MEMORY_PROFILER_MAX_ENTRIES", 100000),
        profile_init=_get_env_bool("VLLM_FL_MEMORY_PROFILER_PROFILE_INIT", True),
        dump_on_exception=_get_env_bool(
            "VLLM_FL_MEMORY_PROFILER_DUMP_ON_EXCEPTION", True
        ),
    )


def get_memory_profiler_dir() -> str:
    return get_memory_profiler_settings().output_dir


def get_memory_profiler_max_entries() -> int:
    return get_memory_profiler_settings().max_entries


def get_memory_profiler_profile_init() -> bool:
    return get_memory_profiler_settings().profile_init


def get_memory_profiler_dump_on_exception() -> bool:
    return get_memory_profiler_settings().dump_on_exception


def memory_profiler_enabled() -> bool:
    return get_memory_profiler_settings().enabled


def _torch_device_fn():
    return current_platform.torch_device_fn


def _memory_module():
    return getattr(_torch_device_fn(), "memory", None)


def _require_memory_snapshot_support() -> Any:
    memory = _memory_module()
    if memory is None or not hasattr(memory, "_record_memory_history"):
        raise RuntimeError(
            "Memory snapshot profiling requires a CUDA-like torch device API "
            "with torch.cuda.memory._record_memory_history support."
        )
    if not hasattr(memory, "_dump_snapshot"):
        raise RuntimeError(
            "Memory snapshot profiling requires a CUDA-like torch device API "
            "with torch.cuda.memory._dump_snapshot support."
        )
    return memory


def _call_if_available(name: str) -> None:
    fn = getattr(_torch_device_fn(), name, None)
    if fn is None:
        return
    try:
        fn()
    except (AttributeError, RuntimeError):
        logger.debug("Ignoring failure from memory profiler device.%s().", name)


class MemorySnapshotProfiler:
    """PyTorch memory snapshot profiler with start/stop API."""

    def __init__(
        self,
        output_dir: str,
        filename_prefix: str = "memory_snapshot",
        max_entries: int = 100000,
        dump_on_exception: bool = True,
    ):
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.max_entries = max_entries
        self.dump_on_exception = dump_on_exception
        self._recording = False
        self._rank: int | None = None

    def set_rank(self, rank: int) -> None:
        self._rank = rank

    def start(self) -> "MemorySnapshotProfiler":
        if self._recording:
            logger.warning("Memory snapshot profiler is already recording.")
            return self

        memory = _require_memory_snapshot_support()
        os.makedirs(self.output_dir, exist_ok=True)

        gc.collect()
        _call_if_available("empty_cache")
        _call_if_available("reset_peak_memory_stats")

        memory._record_memory_history(
            enabled="all",
            stacks="all",
            max_entries=self.max_entries,
        )
        self._recording = True
        logger.info("Memory snapshot profiling started.")
        return self

    def _make_snapshot_path(self, suffix: str | None = None) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        parts = [self.filename_prefix]
        if self._rank is not None:
            parts.append(f"rank{self._rank}")
        if suffix:
            parts.append(suffix)
        parts.append(timestamp)
        filename = "_".join(parts) + ".pickle"
        return os.path.join(self.output_dir, filename)

    def stop(self, suffix: str | None = None) -> str | None:
        if not self._recording:
            logger.warning("Memory snapshot profiler is not recording.")
            return None

        memory = _require_memory_snapshot_support()
        _call_if_available("synchronize")

        snapshot_file = self._make_snapshot_path(suffix=suffix)
        try:
            memory._dump_snapshot(snapshot_file)
        finally:
            memory._record_memory_history(enabled=None)
            self._recording = False

        logger.info(
            "Memory snapshot saved to %s. Visualize at https://pytorch.org/memory_viz",
            snapshot_file,
        )
        return snapshot_file

    def dump_on_error(self) -> str | None:
        if not self._recording:
            return None

        try:
            memory = _require_memory_snapshot_support()
            _call_if_available("synchronize")
            snapshot_file = self._make_snapshot_path(suffix="error")
            memory._dump_snapshot(snapshot_file)
            logger.info(
                "Memory snapshot (error) saved to %s. "
                "Visualize at https://pytorch.org/memory_viz",
                snapshot_file,
            )
            return snapshot_file
        except Exception as exc:
            logger.warning("Failed to dump memory snapshot on error: %s", exc)
            return None

    def __enter__(self) -> "MemorySnapshotProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None and self.dump_on_exception:
            self.dump_on_error()
        self.stop()

    @property
    def is_recording(self) -> bool:
        return self._recording


def memory_snapshot_context(
    rank: int,
    stage: str,
):
    settings = get_memory_profiler_settings()
    if not settings.enabled or not settings.profile_init:
        return nullcontext()

    profiler = MemorySnapshotProfiler(
        output_dir=settings.output_dir,
        filename_prefix=stage,
        max_entries=settings.max_entries,
        dump_on_exception=settings.dump_on_exception,
    )
    profiler.set_rank(rank)
    return profiler
