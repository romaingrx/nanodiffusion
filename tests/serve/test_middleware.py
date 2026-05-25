"""Unit tests for the pure-ASGI request-context middleware."""

import asyncio

import structlog

from nanodiffusion.serve.middleware import RequestContextMiddleware


class _CapturingApp:
    """Downstream ASGI app that snapshots the structlog contextvars on call."""

    def __init__(self) -> None:
        self.captured: dict[str, object] = {}

    async def __call__(self, _scope: object, _receive: object, send: object) -> None:
        self.captured = dict(structlog.contextvars.get_contextvars())
        assert callable(send)
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


async def _receive() -> dict[str, object]:
    return {"type": "http.request", "body": b"", "more_body": False}


async def _noop_send(_message: dict[str, object]) -> None:
    return None


def _drive(app: _CapturingApp, scope: dict[str, object]) -> _CapturingApp:
    middleware = RequestContextMiddleware(app)
    asyncio.run(middleware(scope, _receive, _noop_send))
    return app


def test_middleware_binds_request_scope_into_contextvars() -> None:
    scope = {"type": "http", "method": "POST", "path": "/api/chat"}
    app = _drive(_CapturingApp(), scope)

    assert app.captured["method"] == "POST"
    assert app.captured["path"] == "/api/chat"
    assert isinstance(app.captured["request_id"], str)
    assert len(app.captured["request_id"]) == 8


def test_middleware_skips_non_http_scopes() -> None:
    scope = {"type": "lifespan"}
    app = _drive(_CapturingApp(), scope)

    assert app.captured == {}


def test_middleware_clears_contextvars_before_binding() -> None:
    structlog.contextvars.bind_contextvars(stale="from-previous-request")
    scope = {"type": "http", "method": "GET", "path": "/api/health"}
    app = _drive(_CapturingApp(), scope)

    assert "stale" not in app.captured
