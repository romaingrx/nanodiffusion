"""Tests for the async metrics reporter and its sinks."""

import json
from functools import partial
from pathlib import Path

from nanodiffusion.reporter import (
    InlineReporter,
    JsonlSink,
    MetricEvent,
    MetricSink,
    Reporter,
    StructlogSink,
)


class _MemorySink:
    """In-process sink that records every event for test assertions."""

    def __init__(self, records: list[MetricEvent]) -> None:
        self._records = records

    def log(self, event: MetricEvent) -> None:
        self._records.append(event)

    def close(self) -> None:
        pass


def test_inline_reporter_dispatches_to_every_sink(tmp_path: Path) -> None:
    recorded: list[MetricEvent] = []
    jsonl = tmp_path / "metrics.jsonl"

    with InlineReporter(
        [
            partial(_MemorySink, recorded),
            partial(JsonlSink, jsonl),
        ]
    ) as r:
        r.log(step=1, metrics={"loss": 3.14, "grad_norm": 0.5})
        r.log(step=2, metrics={"loss": 2.71, "grad_norm": 0.4, "lr": 0.001})

    assert len(recorded) == 2
    assert recorded[0].step == 1
    assert recorded[0].metrics["loss"] == 3.14
    assert recorded[1].metrics["lr"] == 0.001

    lines = jsonl.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["step"] == 1
    assert first["loss"] == 3.14
    assert first["grad_norm"] == 0.5
    assert "wall_time" in first


def test_inline_reporter_survives_sink_close_failure() -> None:
    """A raising ``close()`` on one sink must not prevent the rest from closing."""
    closed: list[str] = []

    class BadSink:
        def log(self, event: MetricEvent) -> None:
            pass

        def close(self) -> None:
            msg = "boom"
            raise RuntimeError(msg)

    class GoodSink:
        def log(self, event: MetricEvent) -> None:
            pass

        def close(self) -> None:
            closed.append("good")

    with InlineReporter([BadSink, GoodSink]) as r:
        r.log(step=1, metrics={"loss": 1.0})

    assert "good" in closed


def test_jsonl_sink_appends_and_line_buffers(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "metrics.jsonl"
    sink = JsonlSink(path)
    sink.log(MetricEvent(step=1, metrics={"loss": 1.0}, wall_time=123.0))
    sink.log(MetricEvent(step=2, metrics={"loss": 0.5}, wall_time=124.0))
    sink.close()

    assert path.exists()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0] == {"step": 1, "wall_time": 123.0, "loss": 1.0}
    assert rows[1] == {"step": 2, "wall_time": 124.0, "loss": 0.5}


def test_structlog_sink_is_log_only_and_close_is_noop() -> None:
    sink = StructlogSink("train")
    sink.log(MetricEvent(step=1, metrics={"loss": 1.0}, wall_time=0.0))
    sink.close()


def test_reporter_async_worker_forwards_events_to_jsonl(tmp_path: Path) -> None:
    """Smoke test for the spawned-worker path via a JsonlSink.

    ``functools.partial(JsonlSink, path)`` is picklable (both
    ``JsonlSink`` and ``Path`` survive pickle) so it can safely cross
    the spawn boundary. Tests must never use ``lambda`` factories with
    the async Reporter for this reason.
    """
    path = tmp_path / "async_metrics.jsonl"
    with Reporter([partial(JsonlSink, path)], join_timeout=15.0) as r:
        r.log(step=1, metrics={"loss": 1.5})
        r.log(step=2, metrics={"loss": 1.2, "grad_norm": 0.3})

    # On context exit the worker drains the queue and closes the sink,
    # so the file is fully flushed by the time we read it.
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["step"] == 1
    assert rows[0]["loss"] == 1.5
    assert rows[1]["grad_norm"] == 0.3


def test_metric_sink_protocol_accepts_duck_typed_classes() -> None:
    """Protocol runtime check: any class with ``log`` + ``close`` satisfies it."""
    recorded: list[MetricEvent] = []
    sink = _MemorySink(recorded)
    assert isinstance(sink, MetricSink)
