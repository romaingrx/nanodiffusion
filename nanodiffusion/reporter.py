"""Async metrics reporting with pluggable sinks.

The training loop only ever holds a small in-memory queue; a spawned
worker process constructs the configured sinks and drains the queue so
slow network sinks (wandb) can never block an H100 at $24/hr. Sinks are
created via zero-arg factories because wandb run objects and open file
handles are not picklable across process boundaries, and the async
Reporter needs to pickle its factories across to the worker; that
constrains real callers to use module-level callables or
``functools.partial`` wrappers. The :class:`InlineReporter` variant
stays in-process and has no such restriction, which is what the tests
and tiny local runs use.
"""

import dataclasses
import json
import multiprocessing as mp
import queue
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from multiprocessing.context import (
    SpawnProcess,  # noqa: TC003 - used in class attr annotation evaluated at init
)
from multiprocessing.queues import Queue as MPQueue
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)


type MetricValue = float | int | str
type Metrics = Mapping[str, MetricValue]
type SinkFactory = Callable[[], "MetricSink"]


@dataclasses.dataclass(frozen=True, slots=True)
class MetricEvent:
    """One metric point: which step, what values, when captured."""

    step: int
    metrics: dict[str, MetricValue]
    wall_time: float


@runtime_checkable
class MetricSink(Protocol):
    """A destination for :class:`MetricEvent`s.

    Instances live in the worker process under the async Reporter and
    in the caller process under :class:`InlineReporter`.
    """

    def log(self, event: MetricEvent) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class MetricReporter(Protocol):
    """The interface the training loop calls into.

    Both :class:`Reporter` and :class:`InlineReporter` satisfy this so
    the loop can remain agnostic to whether sinks run in-process or in
    a worker.
    """

    def log(self, step: int, metrics: Metrics) -> None: ...

    def close(self) -> None: ...


class StructlogSink:
    """Structlog sink emitting under a fixed event name.

    Mirrors the shape of the pre-reporter ``logger.info(event_name, ...)``
    site so runs without wandb look identical to before.
    """

    def __init__(self, event_name: str) -> None:
        self._event = event_name
        self._log = structlog.get_logger(__name__)

    def log(self, event: MetricEvent) -> None:
        self._log.info(self._event, step=event.step, **event.metrics)

    def close(self) -> None:
        pass


class JsonlSink:
    """Line-buffered JSONL sink.

    Flushes every line so a crashed run still leaves a parseable log.
    Opened in the sink's own process to avoid cross-process fd sharing.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1)

    def log(self, event: MetricEvent) -> None:
        row: dict[str, Any] = {
            "step": event.step,
            "wall_time": event.wall_time,
            **event.metrics,
        }
        self._fh.write(json.dumps(row) + "\n")

    def close(self) -> None:
        self._fh.close()


class WandbSink:
    """Wandb sink that owns a ``wandb.init`` run.

    Imports ``wandb`` lazily in the constructor so only the sink's own
    process pays for the import and owns the run handle. If the worker
    is under the async Reporter the training process never imports
    wandb at all.
    """

    def __init__(
        self,
        *,
        project: str,
        run_name: str | None,
        config: Mapping[str, Any],
        entity: str | None = None,
    ) -> None:
        import wandb  # noqa: PLC0415

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=dict(config),
            reinit=True,
        )

    def log(self, event: MetricEvent) -> None:
        self._wandb.log(dict(event.metrics), step=event.step)

    def close(self) -> None:
        self._run.finish()


class Reporter(AbstractContextManager["Reporter"]):
    """Fan-out metrics emitter backed by a worker process.

    The training process only calls :meth:`log` which does a
    non-blocking ``put_nowait`` on a bounded queue; a spawned worker
    drains the queue and forwards each event to every configured sink.
    On queue saturation the event is dropped and a warning is emitted;
    the training loop must never block on I/O.

    ``spawn`` is used because ``fork`` is unsafe once JAX/XLA has
    initialised its device backends. The cost is a one-time ~1s worker
    startup, which is irrelevant for multi-hour training runs.
    """

    def __init__(
        self,
        sink_factories: Sequence[SinkFactory],
        *,
        max_queue: int = 1024,
        join_timeout: float = 30.0,
    ) -> None:
        self._factories = tuple(sink_factories)
        self._max_queue = max_queue
        self._join_timeout = join_timeout
        self._ctx = mp.get_context("spawn")
        self._queue: MPQueue[MetricEvent | None] | None = None
        self._proc: SpawnProcess | None = None

    def __enter__(self) -> "Reporter":  # noqa: PYI034 (beartype can't handle PEP 673 Self)
        self._queue = self._ctx.Queue(maxsize=self._max_queue)
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(self._factories, self._queue),
            daemon=True,
        )
        self._proc.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def log(self, step: int, metrics: Metrics) -> None:
        if self._queue is None:
            msg = "Reporter.log called outside its context manager"
            raise RuntimeError(msg)
        event = MetricEvent(step=step, metrics=dict(metrics), wall_time=time.time())
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("metric_queue_full_dropping", step=step)

    def close(self) -> None:
        if self._queue is None or self._proc is None:
            return
        try:
            self._queue.put(None, timeout=self._join_timeout)
        except queue.Full:
            logger.warning("reporter_close_queue_full")
        self._proc.join(timeout=self._join_timeout)
        if self._proc.is_alive():
            logger.warning("reporter_worker_timeout_terminating")
            self._proc.terminate()
            self._proc.join(timeout=5.0)
        self._queue = None
        self._proc = None


class InlineReporter(AbstractContextManager["InlineReporter"]):
    """Synchronous in-process Reporter for tests and tiny runs.

    Exposes the same :meth:`log` surface as :class:`Reporter`, but sinks
    are built in the caller process and events are dispatched inline.
    Tests use this to assert on sink state without a subprocess, and
    small local runs can use it to avoid the spawn overhead.
    """

    def __init__(self, sink_factories: Sequence[SinkFactory]) -> None:
        self._factories = tuple(sink_factories)
        self._sinks: list[MetricSink] = []

    def __enter__(self) -> "InlineReporter":  # noqa: PYI034 (beartype can't handle PEP 673 Self)
        self._sinks = [f() for f in self._factories]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def log(self, step: int, metrics: Metrics) -> None:
        event = MetricEvent(step=step, metrics=dict(metrics), wall_time=time.time())
        for sink in self._sinks:
            sink.log(event)

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                logger.exception("sink_close_failed", sink=type(sink).__name__)
        self._sinks = []


def _worker_main(
    factories: Sequence[SinkFactory],
    q: MPQueue[MetricEvent | None],
) -> None:
    """Worker entry: construct sinks, drain the queue, close on sentinel.

    Sink failures are logged and swallowed so one broken sink cannot
    kill the worker and silently stop metric forwarding for the other
    sinks.
    """
    sinks: list[MetricSink] = []
    for f in factories:
        try:
            sinks.append(f())
        except Exception:
            logger.exception("sink_init_failed")
    try:
        while True:
            event = q.get()
            if event is None:
                break
            for sink in sinks:
                try:
                    sink.log(event)
                except Exception:
                    logger.exception("sink_log_failed", sink=type(sink).__name__)
    finally:
        for sink in sinks:
            try:
                sink.close()
            except Exception:
                logger.exception("sink_close_failed", sink=type(sink).__name__)
