"""FastAPI layer: lifespan-scoped model loading + route marshalling."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from nanodiffusion.inference import (
    Runtime,
    SampleConfigOverride,
    load_runtime,
    warmup,
)
from nanodiffusion.serve.generation import generate_blocking, generate_stream
from nanodiffusion.serve.protocol import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StreamFrame,
)

log = structlog.get_logger(__name__)


async def _sse_events(
    frames: Iterator[StreamFrame], request: Request
) -> AsyncIterator[dict[str, str]]:
    """Bridge a blocking sampler iterator to SSE.

    Each ``next(frames)`` is offloaded to a thread so the event loop
    stays free to detect client disconnects mid-generation.
    StopIteration is coerced to ``None`` because async generators
    cannot let it bubble (CPython raises :class:`RuntimeError`).
    """

    def _next_or_none() -> StreamFrame | None:
        try:
            return next(frames)
        except StopIteration:
            return None

    while True:
        if await request.is_disconnected():
            return
        frame = await asyncio.to_thread(_next_or_none)
        if frame is None:
            return
        yield {"data": frame.model_dump_json(), "id": str(frame.step)}


def create_app(
    *,
    checkpoint: Path,
    overrides: SampleConfigOverride | None = None,
) -> FastAPI:
    overrides = overrides if overrides is not None else SampleConfigOverride()

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

    @app.post("/api/chat/stream")
    async def chat_stream(  # pyright: ignore[reportUnusedFunction]
        req: ChatRequest,
        request: Request,
        rt: Annotated[Runtime, Depends(_runtime)],
    ) -> EventSourceResponse:
        try:
            frames = generate_stream(rt, req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return EventSourceResponse(_sse_events(frames, request))

    return app
