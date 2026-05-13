# Copyright (c) 2025 BAAI. All rights reserved.
#
# FL memory snapshot profiler integration for vLLM versions without the
# upstream memory profiling RPC/API plumbing.
# SPDX-License-Identifier: Apache-2.0

from vllm.logger import init_logger

from vllm_fl.worker.memory_snapshot_profiler import (
    get_memory_profiler_dir,
    memory_profiler_enabled,
)

logger = init_logger(__name__)

_PATCHED = False
_PROFILE_ROUTER_PATCHED = False


def _patch_engine_core_client() -> None:
    try:
        from vllm.v1.engine import core_client
    except Exception:
        return

    engine_core_client = getattr(core_client, "EngineCoreClient", None)
    if engine_core_client is not None and not hasattr(engine_core_client, "mem_profile"):

        def mem_profile(self, is_start: bool = True) -> None:
            raise NotImplementedError

        async def mem_profile_async(self, is_start: bool = True) -> None:
            raise NotImplementedError

        engine_core_client.mem_profile = mem_profile
        engine_core_client.mem_profile_async = mem_profile_async

    inproc_client = getattr(core_client, "InprocClient", None)
    if inproc_client is not None and "mem_profile" not in inproc_client.__dict__:

        def mem_profile(self, is_start: bool = True) -> None:
            self.engine_core.mem_profile(is_start)

        inproc_client.mem_profile = mem_profile

    sync_mp_client = getattr(core_client, "SyncMPClient", None)
    if sync_mp_client is not None and "mem_profile" not in sync_mp_client.__dict__:

        def mem_profile(self, is_start: bool = True) -> None:
            self.call_utility("mem_profile", is_start)

        sync_mp_client.mem_profile = mem_profile

    async_mp_client = getattr(core_client, "AsyncMPClient", None)
    if async_mp_client is not None and "mem_profile_async" not in async_mp_client.__dict__:

        async def mem_profile_async(self, is_start: bool = True) -> None:
            await self.call_utility_async("mem_profile", is_start)

        async_mp_client.mem_profile_async = mem_profile_async


def _patch_engine_core() -> None:
    try:
        from vllm.v1.engine.core import EngineCore
    except Exception:
        return

    if hasattr(EngineCore, "mem_profile"):
        return

    def mem_profile(self, is_start: bool = True) -> None:
        self.model_executor.mem_profile(is_start)

    EngineCore.mem_profile = mem_profile


def _patch_executor() -> None:
    try:
        from vllm.v1.executor.abstract import Executor
    except Exception:
        return

    if hasattr(Executor, "mem_profile"):
        return

    def mem_profile(self, is_start: bool = True) -> None:
        self.collective_rpc("mem_profile", args=(is_start,))

    Executor.mem_profile = mem_profile


def _patch_async_llm() -> None:
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception:
        return

    if "start_mem_profile" not in AsyncLLM.__dict__:

        async def start_mem_profile(self) -> None:
            await self.engine_core.mem_profile_async(True)

        AsyncLLM.start_mem_profile = start_mem_profile

    if "stop_mem_profile" not in AsyncLLM.__dict__:

        async def stop_mem_profile(self) -> None:
            await self.engine_core.mem_profile_async(False)

        AsyncLLM.stop_mem_profile = stop_mem_profile


def _patch_profile_router() -> bool:
    try:
        from fastapi import APIRouter, Request
        from fastapi.responses import Response
        from vllm.entrypoints.serve.profile import api_router
    except Exception:
        return False

    if getattr(api_router.attach_router, "_vllm_fl_memory_profile_patched", False):
        return True

    fl_router = APIRouter()

    @fl_router.post("/start_mem_profile")
    async def start_mem_profile(raw_request: Request):
        logger.info("Starting FL memory snapshot profiler...")
        await api_router.engine_client(raw_request).start_mem_profile()
        logger.info("FL memory snapshot profiler started.")
        return Response(status_code=200)

    @fl_router.post("/stop_mem_profile")
    async def stop_mem_profile(raw_request: Request):
        logger.info("Stopping FL memory snapshot profiler...")
        await api_router.engine_client(raw_request).stop_mem_profile()
        logger.info("FL memory snapshot profiler stopped. Snapshots saved.")
        return Response(status_code=200)

    original_attach_router = api_router.attach_router

    def attach_router(app):  # type: ignore[no-untyped-def]
        original_attach_router(app)
        if memory_profiler_enabled():
            if getattr(app.state, "_vllm_fl_memory_profile_router_attached", False):
                return
            logger.warning_once(
                "FL memory snapshot profiler is enabled in the API server. "
                "Snapshots will be saved to '%s'. "
                "This should ONLY be used for local development!",
                get_memory_profiler_dir(),
            )
            app.include_router(fl_router)
            app.state._vllm_fl_memory_profile_router_attached = True
            return

    attach_router._vllm_fl_memory_profile_patched = True
    api_router.attach_router = attach_router
    return True


def _patch_engine_protocol() -> None:
    try:
        from vllm.engine.protocol import EngineClient
    except Exception:
        return

    if not hasattr(EngineClient, "start_mem_profile"):

        async def start_mem_profile(self) -> None:
            raise NotImplementedError

        EngineClient.start_mem_profile = start_mem_profile

    if not hasattr(EngineClient, "stop_mem_profile"):

        async def stop_mem_profile(self) -> None:
            raise NotImplementedError

        EngineClient.stop_mem_profile = stop_mem_profile


def apply_memory_profile_patches() -> None:
    global _PATCHED, _PROFILE_ROUTER_PATCHED

    # All patch functions are idempotent (they check hasattr internally),
    # so it's safe to retry them. This handles circular imports that cause
    # silent failures during early plugin registration.
    _patch_executor()
    _patch_engine_core()
    _patch_engine_core_client()
    _patch_engine_protocol()
    _patch_async_llm()
    _PATCHED = True

    if not _PROFILE_ROUTER_PATCHED:
        _PROFILE_ROUTER_PATCHED = _patch_profile_router()
