# SPDX-License-Identifier: Apache-2.0
"""Graph capture lifecycle tests using CPU mocks for device operations."""

import gc
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("device_type", ["cuda", "npu", "musa"])
@pytest.mark.parametrize("fail_capture", [False, True])
def test_capture_context_lifetime(monkeypatch, device_type, fail_capture):
    import vllm.model_executor.offloader.base as offloader

    import vllm_fl.compilation.graph as graph
    from vllm_fl.platform import PlatformFL

    gc_collect = Mock()
    cuda_empty = Mock()
    platform_empty = Mock()
    monkeypatch.setattr(gc, "collect", gc_collect)
    monkeypatch.setattr(torch.cuda, "empty_cache", cuda_empty)
    monkeypatch.setattr(PlatformFL, "empty_cache", platform_empty)
    monkeypatch.setattr(offloader, "get_offloader", Mock(return_value=Mock()))
    captured = Mock()
    stream = object()
    output = object()

    @contextmanager
    def capture_context(actual_graph, **kwargs):
        assert actual_graph is captured
        assert kwargs["pool"] == "pool"
        assert ("stream" in kwargs) == (device_type == "cuda")
        if device_type == "cuda":
            assert kwargs["stream"] is stream
            torch.cuda.empty_cache()
            cuda_empty.assert_not_called()
        gc.collect()
        PlatformFL.empty_cache()
        gc_collect.assert_not_called()
        platform_empty.assert_not_called()
        yield

    platform = SimpleNamespace(
        device_type=device_type,
        get_global_graph_pool=lambda: "pool",
        torch_device_fn=SimpleNamespace(graph=capture_context, synchronize=Mock()),
    )
    monkeypatch.setattr(graph, "current_platform", platform)
    monkeypatch.setattr(graph, "current_stream", lambda: stream)
    monkeypatch.setattr(graph, "Graph", SimpleNamespace(graph=lambda: captured))
    monkeypatch.setattr(graph, "set_graph_pool_id", Mock())
    monkeypatch.setattr(graph, "weak_ref_tensors", lambda value: value)
    monkeypatch.setattr(graph, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(graph, "validate_cudagraph_capturing_enabled", Mock())
    monkeypatch.setattr(
        graph,
        "get_forward_context",
        lambda: SimpleNamespace(
            batch_descriptor="batch", cudagraph_runtime_mode=graph.CUDAGraphMode.FULL
        ),
    )
    runnable = Mock(side_effect=RuntimeError("capture failed")) if fail_capture else Mock(return_value=output)
    wrapper = graph.GraphWrapper(
        runnable,
        SimpleNamespace(compilation_config=object()),
        graph.CUDAGraphMode.FULL,
        graph.GraphOptions(gc_disable=True, weak_ref_output=False),
    )
    if fail_capture:
        with pytest.raises(RuntimeError, match="capture failed"):
            wrapper()
        assert wrapper.concrete_graph_entries["batch"].graph is None
    else:
        assert wrapper() is output
        assert wrapper() is output
        runnable.assert_called_once()
        captured.replay.assert_called_once()
        assert platform.torch_device_fn.synchronize.call_count == (device_type == "npu")

    # Both normal return and exceptions must restore process-wide functions.
    assert gc.collect is gc_collect
    assert torch.cuda.empty_cache is cuda_empty
    assert PlatformFL.empty_cache is platform_empty
