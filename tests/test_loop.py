from __future__ import annotations

import signal
import time
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import optax
import structlog.testing

from nanodiffusion.checkpoint import flush, make_manager
from nanodiffusion.loop import LoopState, _save_with_logging
from nanodiffusion.model import Transformer
from nanodiffusion.signals import install_stop_handlers

if TYPE_CHECKING:
    from pathlib import Path

    from nanodiffusion.config import ModelConfig


def test_stop_handler_records_sigterm_and_restores_previous_handler() -> None:
    previous = signal.getsignal(signal.SIGTERM)
    with install_stop_handlers() as stop:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        typed_handler = handler
        typed_handler(signal.SIGTERM, None)
        assert stop.signum == signal.SIGTERM
        assert stop.requested

    assert signal.getsignal(signal.SIGTERM) == previous


def test_save_with_logging_emits_submit_and_finalize_events(
    tmp_path: Path,
    small_config: ModelConfig,
    key: jax.Array,
) -> None:
    """``_save_with_logging`` emits submit + finalize events with throughput math."""
    model = Transformer(small_config, key=key)
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    state = LoopState(
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        step=1,
        cursor=None,
    )

    mngr = make_manager(tmp_path)
    with structlog.testing.capture_logs() as cap_logs:
        _save_with_logging(state, mngr=mngr, ckpt_uri=str(tmp_path))
        flush(mngr)
        # The finalize daemon races with flush; give it a moment to log.
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline and not any(
            r.get("event") == "checkpoint_save_finalized" for r in cap_logs
        ):
            time.sleep(0.01)

    submit = next(r for r in cap_logs if r.get("event") == "checkpoint_save_submitted")
    final = next(r for r in cap_logs if r.get("event") == "checkpoint_save_finalized")
    assert submit["step"] == 1
    assert submit["bytes_est"] > 0
    assert final["step"] == 1
    assert final["bytes_est"] == submit["bytes_est"]
    assert final["wall_s"] > 0
    assert final["throughput_mb_s"] > 0
