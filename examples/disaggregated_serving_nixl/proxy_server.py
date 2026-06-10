# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

"""A lightweight NIXL P/D proxy for vLLM disaggregated serving.

This proxy coordinates one request across a prefill vLLM instance and a decode
vLLM instance:

1. Send a short max_tokens=1 request to the prefiller to produce KV cache.
2. Stream the original request from the decoder with KV-transfer params pointing
   at the selected prefiller side-channel.

It is designed to be more operationally useful than vLLM's test-only
toy_proxy_server.py while still staying small enough to run as a standalone
script. For hardened production deployments, put it behind an ingress/reverse
proxy that handles auth, TLS, and rate limiting.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

logger = logging.getLogger("nixl_pd_proxy")
args: argparse.Namespace


@dataclass
class Backend:
    role: str
    url: str
    side_channel_host: str | None = None
    side_channel_port: int | None = None
    healthy: bool = False
    in_flight: int = 0
    successes: int = 0
    failures: int = 0
    last_error: str = ""
    client: httpx.AsyncClient | None = None

    def mark_success(self) -> None:
        self.successes += 1
        self.last_error = ""

    def mark_failure(self, exc: BaseException) -> None:
        self.failures += 1
        self.last_error = str(exc)


class RoundRobinPool:
    def __init__(self, backends: list[Backend]) -> None:
        self.backends = backends
        self._iterator = itertools.cycle(range(len(backends)))
        self._lock = asyncio.Lock()

    async def pick(self) -> Backend:
        async with self._lock:
            for _ in range(len(self.backends)):
                backend = self.backends[next(self._iterator)]
                if backend.healthy:
                    return backend

            # Keep the service usable during startup or transient health-check
            # jitter. The request path will still surface real backend errors.
            if self.backends:
                return self.backends[next(self._iterator)]

        raise HTTPException(status_code=503, detail="No backend instances configured")


class ProxyState:
    def __init__(self) -> None:
        self.prefillers: list[Backend] = []
        self.decoders: list[Backend] = []
        self.prefill_pool: RoundRobinPool | None = None
        self.decode_pool: RoundRobinPool | None = None
        self.ready = asyncio.Event()
        self.shutdown = asyncio.Event()
        self.semaphore: asyncio.Semaphore | None = None
        self.started_at = time.time()
        self.requests_total = 0
        self.requests_failed = 0


state = ProxyState()


def _build_url(host: str, port: int) -> str:
    if host.startswith("http://") or host.startswith("https://"):
        return f"{host.rstrip('/')}:{port}" if ":" not in host.rsplit("/", 1)[-1] else host.rstrip("/")
    return f"http://{host}:{port}"


def _forward_headers(request: Request, request_id: str) -> dict[str, str]:
    headers: dict[str, str] = {"X-Request-Id": request_id}
    for key, value in request.headers.items():
        lowered = key.lower()
        if lowered in HOP_BY_HOP_HEADERS:
            continue
        if lowered == "x-request-id":
            continue
        headers[key] = value
    return headers


async def _health_check_loop(backend: Backend) -> None:
    assert backend.client is not None
    while not state.shutdown.is_set():
        try:
            response = await backend.client.get("/health")
            response.raise_for_status()
            if not backend.healthy:
                logger.info("%s backend healthy: %s", backend.role, backend.url)
            backend.healthy = True
            backend.last_error = ""
        except Exception as exc:
            if backend.healthy:
                logger.warning(
                    "%s backend unhealthy: %s error=%s",
                    backend.role,
                    backend.url,
                    exc,
                )
            backend.healthy = False
            backend.last_error = str(exc)
        await asyncio.sleep(args.health_interval)


async def _wait_for_initial_health() -> None:
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if any(b.healthy for b in state.prefillers) and any(b.healthy for b in state.decoders):
            state.ready.set()
            return
        await asyncio.sleep(0.5)
    logger.warning(
        "Startup health timeout reached; accepting requests and surfacing backend errors."
    )
    state.ready.set()


def _make_prefill_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(request_payload)
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
    }
    payload["stream"] = False
    payload["max_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)
    return payload


def _make_decode_payload(
    request_payload: dict[str, Any],
    prefiller: Backend,
) -> dict[str, Any]:
    if prefiller.side_channel_host is None or prefiller.side_channel_port is None:
        raise HTTPException(
            status_code=500,
            detail=f"Prefiller side-channel is not configured for {prefiller.url}",
        )
    payload = dict(request_payload)
    payload["kv_transfer_params"] = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_host": prefiller.side_channel_host,
        "remote_port": prefiller.side_channel_port,
    }
    return payload


async def _send_prefill(
    prefiller: Backend,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> None:
    assert prefiller.client is not None
    last_exc: BaseException | None = None
    prefiller.in_flight += 1
    try:
        for attempt in range(args.prefill_retries + 1):
            try:
                response = await prefiller.client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=args.prefill_timeout,
                )
                response.raise_for_status()
                await response.aclose()
                prefiller.mark_success()
                return
            except Exception as exc:
                last_exc = exc
                prefiller.mark_failure(exc)
                if attempt < args.prefill_retries:
                    await asyncio.sleep(args.retry_delay)
        assert last_exc is not None
        raise last_exc
    finally:
        prefiller.in_flight -= 1


async def _stream_decode(
    decoder: Backend,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
):
    assert decoder.client is not None
    decoder.in_flight += 1
    try:
        async with decoder.client.stream(
            "POST",
            endpoint,
            json=payload,
            headers=headers,
            timeout=args.decode_timeout,
        ) as response:
            response.raise_for_status()
            decoder.mark_success()
            async for chunk in response.aiter_bytes():
                yield chunk
    except Exception as exc:
        decoder.mark_failure(exc)
        raise
    finally:
        decoder.in_flight -= 1


async def _handle_openai_request(endpoint: str, request: Request):
    if not state.ready.is_set():
        raise HTTPException(status_code=503, detail="Proxy is starting")
    if state.semaphore is None or state.prefill_pool is None or state.decode_pool is None:
        raise HTTPException(status_code=503, detail="Proxy is not initialized")

    async with state.semaphore:
        state.requests_total += 1
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        try:
            request_payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

        prefiller = await state.prefill_pool.pick()
        decoder = await state.decode_pool.pick()
        headers = _forward_headers(request, request_id)
        prefill_payload = _make_prefill_payload(request_payload)
        decode_payload = _make_decode_payload(request_payload, prefiller)

        logger.info(
            "request start id=%s endpoint=%s prefiller=%s decoder=%s",
            request_id,
            endpoint,
            prefiller.url,
            decoder.url,
        )

        prefill_task = asyncio.create_task(
            _send_prefill(prefiller, endpoint, prefill_payload, headers)
        )

        if args.prefill_mode == "sync":
            try:
                await prefill_task
            except Exception as exc:
                state.requests_failed += 1
                logger.exception("prefill failed id=%s", request_id)
                raise HTTPException(status_code=502, detail=f"Prefill failed: {exc}") from exc

        async def generate():
            first_chunk = True
            try:
                async for chunk in _stream_decode(decoder, endpoint, decode_payload, headers):
                    first_chunk = False
                    yield chunk
                if not prefill_task.done():
                    await prefill_task
            except asyncio.CancelledError:
                prefill_task.cancel()
                raise
            except Exception:
                state.requests_failed += 1
                if not prefill_task.done():
                    prefill_task.cancel()
                logger.exception(
                    "request failed id=%s first_chunk_sent=%s",
                    request_id,
                    not first_chunk,
                )
                raise
            finally:
                if prefill_task.done() and not prefill_task.cancelled():
                    exc = prefill_task.exception()
                    if exc is not None:
                        logger.warning("prefill task failed id=%s error=%s", request_id, exc)
                logger.info("request finish id=%s", request_id)

        return StreamingResponse(generate(), media_type="application/json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(args.http_timeout, connect=args.connect_timeout)
    limits = httpx.Limits(
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_keepalive_connections,
    )

    state.semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    for idx, (host, port) in enumerate(zip(args.prefiller_hosts, args.prefiller_ports)):
        side_host = (
            args.prefiller_side_channel_hosts[idx]
            if args.prefiller_side_channel_hosts
            else host
        )
        side_port = (
            args.prefiller_side_channel_ports[idx]
            if args.prefiller_side_channel_ports
            else args.default_side_channel_port
        )
        backend = Backend(
            role="prefill",
            url=_build_url(host, port),
            side_channel_host=side_host.replace("http://", "").replace("https://", ""),
            side_channel_port=side_port,
        )
        backend.client = httpx.AsyncClient(base_url=backend.url, timeout=timeout, limits=limits)
        state.prefillers.append(backend)

    for host, port in zip(args.decoder_hosts, args.decoder_ports):
        backend = Backend(role="decode", url=_build_url(host, port))
        backend.client = httpx.AsyncClient(base_url=backend.url, timeout=timeout, limits=limits)
        state.decoders.append(backend)

    state.prefill_pool = RoundRobinPool(state.prefillers)
    state.decode_pool = RoundRobinPool(state.decoders)

    tasks = [
        asyncio.create_task(_health_check_loop(backend))
        for backend in state.prefillers + state.decoders
    ]
    tasks.append(asyncio.create_task(_wait_for_initial_health()))

    try:
        yield
    finally:
        state.shutdown.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for backend in state.prefillers + state.decoders:
            if backend.client is not None:
                await backend.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    prefill_ok = any(b.healthy for b in state.prefillers)
    decode_ok = any(b.healthy for b in state.decoders)
    status = 200 if prefill_ok and decode_ok else 503
    body = {
        "ok": prefill_ok and decode_ok,
        "prefill_healthy": prefill_ok,
        "decode_healthy": decode_ok,
    }
    return JSONResponse(body, status_code=status)


@app.get("/metrics")
async def metrics():
    def backend_dict(backend: Backend) -> dict[str, Any]:
        return {
            "url": backend.url,
            "healthy": backend.healthy,
            "in_flight": backend.in_flight,
            "successes": backend.successes,
            "failures": backend.failures,
            "last_error": backend.last_error,
            "side_channel_host": backend.side_channel_host,
            "side_channel_port": backend.side_channel_port,
        }

    return {
        "uptime_seconds": time.time() - state.started_at,
        "requests_total": state.requests_total,
        "requests_failed": state.requests_failed,
        "prefillers": [backend_dict(b) for b in state.prefillers],
        "decoders": [backend_dict(b) for b in state.decoders],
    }


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_openai_request("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_openai_request("/v1/chat/completions", request)


def _split_csv_or_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIXL P/D proxy for vLLM")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefiller-hosts", nargs="+", required=True)
    parser.add_argument("--prefiller-ports", nargs="+", type=int, required=True)
    parser.add_argument("--decoder-hosts", nargs="+", required=True)
    parser.add_argument("--decoder-ports", nargs="+", type=int, required=True)
    parser.add_argument("--prefiller-side-channel-hosts", nargs="+")
    parser.add_argument("--prefiller-side-channel-ports", nargs="+", type=int)
    parser.add_argument("--default-side-channel-port", type=int, default=5600)
    parser.add_argument("--prefill-mode", choices=["async", "sync"], default="async")
    parser.add_argument("--prefill-retries", type=int, default=0)
    parser.add_argument("--retry-delay", type=float, default=0.05)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--health-interval", type=float, default=2.0)
    parser.add_argument("--max-concurrent-requests", type=int, default=1024)
    parser.add_argument("--max-connections", type=int, default=2048)
    parser.add_argument("--max-keepalive-connections", type=int, default=512)
    parser.add_argument("--http-timeout", type=float, default=None)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--prefill-timeout", type=float, default=None)
    parser.add_argument("--decode-timeout", type=float, default=None)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))

    parsed = parser.parse_args()
    parsed.prefiller_hosts = _split_csv_or_list(parsed.prefiller_hosts) or []
    parsed.decoder_hosts = _split_csv_or_list(parsed.decoder_hosts) or []
    parsed.prefiller_side_channel_hosts = _split_csv_or_list(
        parsed.prefiller_side_channel_hosts
    )

    if len(parsed.prefiller_hosts) != len(parsed.prefiller_ports):
        parser.error("--prefiller-hosts and --prefiller-ports length mismatch")
    if len(parsed.decoder_hosts) != len(parsed.decoder_ports):
        parser.error("--decoder-hosts and --decoder-ports length mismatch")
    if parsed.prefiller_side_channel_hosts and (
        len(parsed.prefiller_side_channel_hosts) != len(parsed.prefiller_hosts)
    ):
        parser.error("--prefiller-side-channel-hosts length mismatch")
    if parsed.prefiller_side_channel_ports and (
        len(parsed.prefiller_side_channel_ports) != len(parsed.prefiller_hosts)
    ):
        parser.error("--prefiller-side-channel-ports length mismatch")
    return parsed


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
