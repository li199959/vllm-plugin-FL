# Copyright (c) 2025 BAAI. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager

from vllm.v1.worker.gpu_model_runner import (
    AsyncGPUModelRunnerOutput,
    ExecuteModelState,
    GPUModelRunner,
)


@contextmanager
def graph_capture(device):
    from vllm.distributed.parallel_state import graph_capture as vllm_graph_capture

    with vllm_graph_capture(device=device) as ctx:
        yield ctx


class ModelRunnerFL(GPUModelRunner):
    """FL compatibility layer backed by the vLLM 0.16 GPU model runner."""


__all__ = [
    "AsyncGPUModelRunnerOutput",
    "ExecuteModelState",
    "ModelRunnerFL",
    "graph_capture",
]
