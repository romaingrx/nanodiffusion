"""Signal handling helpers for graceful long-running jobs."""

import dataclasses
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

import structlog

logger = structlog.get_logger(__name__)


@dataclasses.dataclass(slots=True)
class StopRequest:
    signum: int | None = None

    @property
    def requested(self) -> bool:
        return self.signum is not None


def signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


@contextmanager
def install_stop_handlers() -> Iterator[StopRequest]:
    """Request graceful stop on SIGINT/SIGTERM without exiting mid-step."""
    stop = StopRequest()
    if threading.current_thread() is not threading.main_thread():
        yield stop
        return

    handled = (signal.SIGINT, signal.SIGTERM)
    previous = {sig: signal.getsignal(sig) for sig in handled}

    def _handler(signum: int, frame: FrameType | None) -> None:
        del frame
        if stop.signum is None:
            stop.signum = signum
            logger.warning("training_stop_requested", signal=signal_name(signum))
            return
        msg = f"received second stop signal {signal_name(signum)}"
        raise KeyboardInterrupt(msg)

    for sig in handled:
        signal.signal(sig, _handler)
    try:
        yield stop
    finally:
        for sig, old_handler in previous.items():
            signal.signal(sig, old_handler)
