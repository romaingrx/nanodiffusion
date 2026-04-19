"""FastAPI layer: lifespan-scoped model loading + route marshalling.

No business logic here. Routes pull the pre-loaded runtime from
``app.state`` and delegate to :mod:`nanodiffusion.serve.generation`,
wrapping blocking calls in ``asyncio.to_thread`` so the event loop
stays free for WebSocket heartbeats during XLA-bound per-step work.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.websockets import WebSocketDisconnect

from nanodiffusion.serve.generation import generate_blocking, generate_stream
from nanodiffusion.serve.protocol import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StreamFrame,
)
from nanodiffusion.serve.runtime import (
    Runtime,
    SampleDefaultsOverride,
    load_runtime,
    warmup,
)

log = structlog.get_logger(__name__)


async def _as_async(gen: Iterator[StreamFrame]) -> AsyncIterator[StreamFrame]:
    """Advance a sync generator through ``asyncio.to_thread`` per step.

    Each ``next(gen)`` blocks on an XLA compute; offloading keeps the
    event loop free for WebSocket heartbeats and cancellation. StopIteration
    is coerced to ``None`` because async generators cannot let it bubble
    (CPython turns it into :class:`RuntimeError`).
    """

    def _next_or_none() -> StreamFrame | None:
        try:
            return next(gen)
        except StopIteration:
            return None

    while True:
        value = await asyncio.to_thread(_next_or_none)
        if value is None:
            return
        yield value


def create_app(
    *,
    checkpoint: Path,
    overrides: SampleDefaultsOverride | None = None,
) -> FastAPI:
    overrides = overrides if overrides is not None else SampleDefaultsOverride()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("serve.loading", checkpoint=str(checkpoint))
        runtime = await asyncio.to_thread(load_runtime, checkpoint, overrides=overrides)
        log.info("serve.warmup_start", max_length=runtime.defaults.max_length)
        await asyncio.to_thread(warmup, runtime)
        log.info("serve.ready", train_step=runtime.train_step)
        app.state.runtime = runtime
        yield

    app = FastAPI(lifespan=lifespan)

    def _runtime(request: Request) -> Runtime:
        rt: Runtime = request.app.state.runtime
        return rt

    @app.get("/api/health")
    def health(  # pyright: ignore[reportUnusedFunction]
        rt: Annotated[Runtime, Depends(_runtime)],
    ) -> HealthResponse:
        return HealthResponse(
            checkpoint=str(rt.checkpoint_path),
            train_step=rt.train_step,
            max_seq_len=rt.max_seq_len,
            vocab_size=rt.tok.vocab_size,
            sample_defaults=rt.defaults,
        )

    @app.post("/api/chat")
    async def chat(  # pyright: ignore[reportUnusedFunction]
        req: ChatRequest, rt: Annotated[Runtime, Depends(_runtime)]
    ) -> ChatResponse:
        try:
            return await asyncio.to_thread(generate_blocking, rt, req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.websocket("/api/chat/stream")
    async def chat_stream(ws: WebSocket) -> None:  # pyright: ignore[reportUnusedFunction]
        await ws.accept()
        rt: Runtime = ws.app.state.runtime
        try:
            payload = await ws.receive_json()
            req = ChatRequest.model_validate(payload)
            gen = generate_stream(rt, req)
            async for frame in _as_async(gen):
                await ws.send_json(frame.model_dump())
        except ValueError as exc:
            await ws.close(code=1008, reason=str(exc)[:120])
            return
        except WebSocketDisconnect:
            return
        await ws.close()

    return app
