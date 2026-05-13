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

_FL_ROUTER = None


def _engine_client(request):  # type: ignore[no-untyped-def]
    return request.app.state.engine_client


def _get_fl_router():
    global _FL_ROUTER
    if _FL_ROUTER is not None:
        return _FL_ROUTER

    from fastapi import APIRouter
    from fastapi.responses import Response

    router = APIRouter()

    @router.post("/start_mem_profile")
    async def start_mem_profile(raw_request):  # type: ignore[no-untyped-def]
        logger.info("Starting FL memory snapshot profiler...")
        await _engine_client(raw_request).start_mem_profile()
        logger.info("FL memory snapshot profiler started.")
        return Response(status_code=200)

    @router.post("/stop_mem_profile")
    async def stop_mem_profile(raw_request):  # type: ignore[no-untyped-def]
        logger.info("Stopping FL memory snapshot profiler...")
        await _engine_client(raw_request).stop_mem_profile()
        logger.info("FL memory snapshot profiler stopped. Snapshots saved.")
        return Response(status_code=200)

    _FL_ROUTER = router
    return router


def _attach_fl_router(app) -> None:  # type: ignore[no-untyped-def]
    if not memory_profiler_enabled():
        return
    if getattr(app.state, "_vllm_fl_memory_profile_router_attached", False):
        return

    logger.warning_once(
        "FL memory snapshot profiler is enabled in the API server. "
        "Snapshots will be saved to '%s'. "
        "This should ONLY be used for local development!",
        get_memory_profiler_dir(),
    )
    app.include_router(_get_fl_router())
    app.state._vllm_fl_memory_profile_router_attached = True
    logger.info(
        "FL memory snapshot profiler route registered: "
        "/start_mem_profile, /stop_mem_profile"
    )


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
    if inproc_client is not None and not hasattr(inproc_client, "mem_profile"):

        def mem_profile(self, is_start: bool = True) -> None:
            self.engine_core.mem_profile(is_start)

        inproc_client.mem_profile = mem_profile

    sync_mp_client = getattr(core_client, "SyncMPClient", None)
    if sync_mp_client is not None and not hasattr(sync_mp_client, "mem_profile"):

        def mem_profile(self, is_start: bool = True) -> None:
            self.call_utility("mem_profile", is_start)

        sync_mp_client.mem_profile = mem_profile

    async_mp_client = getattr(core_client, "AsyncMPClient", None)
    if async_mp_client is not None and not hasattr(async_mp_client, "mem_profile_async"):

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

    if not hasattr(AsyncLLM, "start_mem_profile"):

        async def start_mem_profile(self) -> None:
            await self.engine_core.mem_profile_async(True)

        AsyncLLM.start_mem_profile = start_mem_profile

    if not hasattr(AsyncLLM, "stop_mem_profile"):

        async def stop_mem_profile(self) -> None:
            await self.engine_core.mem_profile_async(False)

        AsyncLLM.stop_mem_profile = stop_mem_profile


def _patch_profile_router() -> None:
    try:
        from vllm.entrypoints.serve.profile import api_router
    except Exception:
        return

    if getattr(api_router.attach_router, "_vllm_fl_memory_profile_patched", False):
        return

    original_attach_router = api_router.attach_router

    def attach_router(app):  # type: ignore[no-untyped-def]
        original_attach_router(app)
        _attach_fl_router(app)

    attach_router._vllm_fl_memory_profile_patched = True
    api_router.attach_router = attach_router


def _patch_openai_build_app() -> None:
    try:
        from vllm.entrypoints.openai import api_server
    except Exception:
        return

    if getattr(api_server.build_app, "_vllm_fl_memory_profile_patched", False):
        return

    original_build_app = api_server.build_app

    def build_app(*args, **kwargs):  # type: ignore[no-untyped-def]
        app = original_build_app(*args, **kwargs)
        _attach_fl_router(app)
        return app

    build_app._vllm_fl_memory_profile_patched = True
    api_server.build_app = build_app


def _patch_serve_router_registration() -> None:
    try:
        import vllm.entrypoints.serve as serve_entrypoints
    except Exception:
        return

    register = serve_entrypoints.register_vllm_serve_api_routers
    if getattr(register, "_vllm_fl_memory_profile_patched", False):
        return

    def register_vllm_serve_api_routers(app):  # type: ignore[no-untyped-def]
        register(app)
        _attach_fl_router(app)

    register_vllm_serve_api_routers._vllm_fl_memory_profile_patched = True
    serve_entrypoints.register_vllm_serve_api_routers = register_vllm_serve_api_routers


def _patch_fastapi_init() -> None:
    try:
        from fastapi import FastAPI
    except Exception:
        return

    if getattr(FastAPI.__init__, "_vllm_fl_memory_profile_patched", False):
        return

    original_init = FastAPI.__init__

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        _attach_fl_router(self)

    __init__._vllm_fl_memory_profile_patched = True
    FastAPI.__init__ = __init__


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
    _patch_executor()
    _patch_engine_core()
    _patch_engine_core_client()
    _patch_engine_protocol()
    _patch_async_llm()
    _patch_profile_router()
    _patch_serve_router_registration()
    _patch_openai_build_app()
    _patch_fastapi_init()
