"""Unified structlog + stdlib logging configuration.

Both :func:`structlog.get_logger` calls and third-party ``logging.getLogger``
calls (uvicorn, fastapi, httpx, ...) flow through the same
:class:`structlog.stdlib.ProcessorFormatter` pipeline, so every log line —
ours or theirs — renders in a consistent shape. ``external_logs=False``
keeps third-party chatter out of the output without touching our own loggers.

Call :func:`configure` exactly once at process entry. Calls after the
first replace the root logger's handlers, so it is safe to re-invoke in
tests.
"""

import enum
import logging
import sys
from typing import TYPE_CHECKING, Final

import structlog

if TYPE_CHECKING:
    from structlog.typing import Processor


class LogFormat(enum.StrEnum):
    """Rendering style for the unified structlog + stdlib pipeline.

    Member names are lowercase so ``click.Choice(LogFormat)`` surfaces
    ``[console|json]`` on the CLI — Click keys on enum *names* for
    user input, not values.
    """

    console = "console"
    json = "json"


_EXTERNAL_LOGGER_NAMES: Final[tuple[str, ...]] = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
    "httpx",
    "httpcore",
    "watchfiles",
)


def configure(
    *,
    level: int = logging.INFO,
    fmt: LogFormat = LogFormat.console,
    external_logs: bool = True,
) -> None:
    """Wire structlog and stdlib ``logging`` to a single formatter.

    Args:
        level: Minimum level for our own loggers and the root logger.
        fmt: ``"console"`` for dev-readable ANSI output, ``"json"`` for
            one-line JSON suitable for log shippers.
        external_logs: When ``False``, loggers in
            :data:`_EXTERNAL_LOGGER_NAMES` are clamped to ``WARNING`` so
            uvicorn access logs and httpx request lines don't crowd the
            stream. Set via ``--no-external-logs`` on the CLI.
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if fmt is LogFormat.json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    external_level = level if external_logs else logging.WARNING
    for name in _EXTERNAL_LOGGER_NAMES:
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
        lg.setLevel(external_level)
