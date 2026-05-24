"""ASGI middleware that binds request-scoped fields into structlog contextvars.

Pure ASGI (not :class:`starlette.middleware.base.BaseHTTPMiddleware`) because
``BaseHTTPMiddleware`` runs the endpoint in a child task group and copies the
context — ``bind_contextvars`` calls inside routes would not propagate back
here, breaking the access log below.
"""

import time
import uuid
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = structlog.get_logger("serve.access")


class RequestContextMiddleware:
    """Scope ``request_id``/``method``/``path`` to the request and emit one
    access line on completion with status and latency.

    Downstream log calls inherit the bound context via
    :func:`structlog.contextvars.merge_contextvars`, which is already in the
    project's processor chain (see :mod:`nanodiffusion.logs`).
    """

    def __init__(self, app: "ASGIApp") -> None:
        self._app = app

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=uuid.uuid4().hex[:8],
            method=scope["method"],
            path=scope["path"],
        )

        status = 500
        start = time.perf_counter()

        async def send_capturing_status(message: "Message") -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self._app(scope, receive, send_capturing_status)
        finally:
            _log_access(status, (time.perf_counter() - start) * 1000)


def _log_access(status: int, duration_ms: float) -> None:
    """Map HTTP status class to a structlog level so 4xx/5xx surface clearly."""
    level = "error" if status >= 500 else "warning" if status >= 400 else "info"  # noqa: PLR2004
    getattr(log, level)(
        "request.complete",
        status=status,
        duration_ms=round(duration_ms, 2),
    )
