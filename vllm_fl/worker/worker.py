# Copyright (c) 2025 BAAI. All rights reserved.

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.utils.mem_utils import MemoryProfilingResult, MemorySnapshot
from vllm.v1.worker.gpu_worker import Worker

import vllm_fl.envs as fl_envs
from vllm_fl.dispatch.io_common import managed_inference_mode
from vllm_fl.ops.custom_ops import register_oot_ops
from vllm_fl.utils import get_flag_gems_whitelist_blacklist

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


@dataclass
class MemoryProfilingResultFL(MemoryProfilingResult):
    """Backward-compatible alias for FL tests."""


@contextmanager
def memory_profiling_fl(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int,
):
    from vllm.utils.mem_utils import memory_profiling

    with memory_profiling(
        baseline_snapshot=baseline_snapshot,
        weights_memory=weights_memory,
    ) as result:
        yield result


class WorkerFL(Worker):
    """FL worker built on top of the vLLM 0.18 GPU worker."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        logger.debug("=== ENVIRONMENT VARIABLES ===")
        for key, value in sorted(os.environ.items()):
            logger.debug("%s=%r", key, value)

        register_oot_ops()
        self._configure_flaggems()

    def _configure_flaggems(self) -> None:
        if not fl_envs.USE_FLAGGEMS:
            return

        try:
            import flag_gems
        except ImportError:
            logger.warning("USE_FLAGGEMS is enabled but flag_gems is not installed.")
            return

        whitelist, blacklist = get_flag_gems_whitelist_blacklist()
        if whitelist:
            logger.info("[FlagGems] Enable only the following ops: %s", whitelist)
            flag_gems.only_enable(
                include=whitelist,
                record=True,
                once=True,
                path=fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH,
            )
            return

        if blacklist:
            logger.info("[FlagGems] Disable the following ops: %s", blacklist)
            flag_gems.enable(
                unused=blacklist,
                record=True,
                once=True,
                path=fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH,
            )
            return

        logger.info("[FlagGems] Enable all ops")
        flag_gems.enable(
            record=True,
            once=True,
            path=fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH,
        )

    @managed_inference_mode()
    def determine_available_memory(self) -> int:
        return super().determine_available_memory()

    @managed_inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput":
        return super().sample_tokens(grammar_output)

    @managed_inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | None":
        return super().execute_model(scheduler_output)


__all__ = [
    "MemoryProfilingResult",
    "MemoryProfilingResultFL",
    "MemorySnapshot",
    "WorkerFL",
    "memory_profiling_fl",
]
