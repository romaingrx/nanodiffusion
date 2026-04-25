"""Shared training loop driving pretrain and SFT.

Paradigm-specific differences (batch pytree shape and optional host-side
extras) are threaded through a :class:`PrepareBatch` callback and the
typed metrics returned by :data:`TrainStepFn`, so
:func:`run_training_loop` owns the timing/saving/metric-forwarding
logic once. The loop never imports wandb or any other metric backend;
it calls into a :class:`~nanodiffusion.reporter.MetricReporter` which
fans out to whatever sinks the caller configured.
"""

import dataclasses
import datetime
import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import jax
import optax
import structlog
from pydantic import BaseModel

from nanodiffusion.checkpoint import save_checkpoint, write_config
from nanodiffusion.data.cursors import LoaderCursor
from nanodiffusion.data.loader import DevicePrefetchIterator
from nanodiffusion.metrics import (
    CoreHostMetrics,
    CoreStepMetrics,
    NoHostExtras,
    ReportMetrics,
    SFTHostExtras,
)
from nanodiffusion.model import DiffusionModel
from nanodiffusion.reporter import MetricReporter
from nanodiffusion.signals import StopRequest, install_stop_handlers, signal_name
from nanodiffusion.train_step import TrainStepFn
from nanodiffusion.types import PRNGKeyArray

logger = structlog.get_logger(__name__)


def make_run_id() -> str:
    """UTC timestamp run id, e.g. ``20260408-193015``."""
    return datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")


def resolve_run_dir(run_dir_root: Path, *, resume_from: Path | None) -> Path:
    """Fresh timestamped run dir, or reuse ``resume_from.parent`` on resume.

    The returned directory exists on disk; callers drop their own
    ``config.yaml`` and ``logger.info`` on top so pretrain and SFT keep
    their event-name conventions separate.
    """
    if resume_from is not None:
        run_dir = resume_from.parent.resolve()
    else:
        run_dir = run_dir_root / make_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclasses.dataclass(frozen=True, slots=True)
class StepStats:
    """Per-step metrics returned by the batch-prepare hook.

    ``supervised_tokens`` is non-zero only for SFT; pretrain leaves it
    at 0 and the loop omits the derived ``supervised_tok_per_s`` field
    from its log output in that case.
    """

    supervised_tokens: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class LoopHyperparams:
    """Paradigm-independent knobs that drive :func:`run_training_loop`."""

    max_steps: int
    log_every: int
    save_every: int
    prefetch_size: int
    nominal_tokens_per_step: int
    event_name: str
    profile_steps: int = 0


@dataclasses.dataclass
class LoopState[M: DiffusionModel, C: LoaderCursor]:
    """Mutable per-iteration state threaded through the loop body.

    Plain Python — not a JAX pytree — so the loop can update
    ``step`` / ``cursor`` / ``last_saved_step`` with normal
    assignment between JIT'd train-step calls.
    """

    model: M
    ema_model: M
    opt_state: optax.OptState
    key: PRNGKeyArray
    step: int
    cursor: C | None
    last_saved_step: int | None = None


type PrepareBatch[B, JB, C: LoaderCursor] = Callable[[B], tuple[JB, C, StepStats]]


def _save_final_checkpoint_if_needed[M: DiffusionModel, C: LoaderCursor](
    state: LoopState[M, C],
    *,
    initial_step: int,
    save: Callable[[], None],
) -> None:
    if state.step > initial_step and state.last_saved_step != state.step:
        save()


def _log_graceful_stop(stop_request: StopRequest, *, step: int, run_dir: Path) -> None:
    if stop_request.signum is None:
        return
    logger.info(
        "training_stopped_gracefully",
        signal=signal_name(stop_request.signum),
        step=step,
        run_dir=str(run_dir),
    )


def _collect_host_metrics(
    step_metrics: CoreStepMetrics,
    lr_schedule: optax.Schedule,
    step: int,
    tok_per_s: int,
    steps_in_window: int,
    nominal_tokens_per_step: int,
    max_steps: int,
    supervised_tokens_in_window: int,
    elapsed: float,
) -> ReportMetrics:
    """Merge device-side step metrics with host-side throughput + HBM.

    Non-finite losses no longer raise. The JIT'd train step skips the
    model update on non-finite gradients (see
    :func:`nanodiffusion.optimizer.apply_or_skip`), so training can
    recover from transient spikes while still surfacing a warning.
    """
    core = CoreHostMetrics.from_step_metrics(
        step_metrics,
        lr_schedule=lr_schedule,
        step=step,
        tok_per_s=tok_per_s,
        steps_per_s=steps_in_window / max(elapsed, 1e-9),
        step_time_ms=elapsed * 1000.0 / max(steps_in_window, 1),
        tokens_seen=step * nominal_tokens_per_step,
        progress_pct=min(100.0, 100.0 * step / max_steps),
    )
    if not math.isfinite(core.loss):
        logger.warning(
            "non_finite_loss",
            step=step,
            loss=core.loss,
            grad_finite=core.grad_finite,
        )
    extras = (
        SFTHostExtras.from_window(
            supervised_tokens_in_window=supervised_tokens_in_window,
            elapsed=elapsed,
        )
        if supervised_tokens_in_window > 0
        else NoHostExtras()
    )
    return ReportMetrics(core=core, extras=extras)


def run_training_loop[M: DiffusionModel, B, JB, C: LoaderCursor](
    state: LoopState[M, C],
    *,
    config: BaseModel,
    run_dir: Path,
    train_step: TrainStepFn[M, JB],
    lr_schedule: optax.Schedule,
    base_loader: Iterator[B],
    settings: LoopHyperparams,
    prepare_batch: PrepareBatch[B, JB, C],
    reporter: MetricReporter,
) -> None:
    """Drive the training loop in place.

    Deliberately not JIT'd: mutates ``state`` via Python assignment
    between JIT'd train-step calls so prefetch, logging, and
    checkpointing interleave with device work. The throughput window is
    reset once after the first post-compile step so JIT compile latency
    is excluded from the first ``tok_per_s`` report; the end-of-training
    save is skipped when a periodic save already landed on the terminal
    step.

    Device placement (replicate / shard) is the caller's responsibility:
    the ``state`` pytrees should already be placed on the target devices,
    and ``prepare_batch`` should shard the batch before returning it.
    This keeps the loop device-agnostic and avoids importing the
    sharding module here.
    """
    initial_step = state.step
    last_log_step = state.step
    supervised_tokens_in_window = 0

    def _save() -> None:
        ckpt_dir = run_dir / f"step_{state.step}"
        save_checkpoint(
            ckpt_dir,
            model=state.model,
            ema_model=state.ema_model,
            opt_state=state.opt_state,
            key=state.key,
            step=state.step,
            cursor=state.cursor,
            update_latest=True,
        )
        write_config(ckpt_dir, config)
        state.last_saved_step = state.step
        logger.info("checkpoint_saved", path=str(ckpt_dir), step=state.step)

    t_window_start = time.monotonic()
    with (
        install_stop_handlers() as stop_request,
        DevicePrefetchIterator(
            base_loader,
            prepare_batch,
            cpu_prefetch=settings.prefetch_size,
            device_prefetch=2,
        ) as loader,
    ):
        for jax_batch, cursor, stats in loader:
            if state.step >= settings.max_steps or stop_request.requested:
                break
            state.cursor = cursor
            supervised_tokens_in_window += stats.supervised_tokens
            (
                state.model,
                state.ema_model,
                state.opt_state,
                step_metrics,
                state.key,
            ) = train_step(
                state.model,
                state.ema_model,
                state.opt_state,
                jax_batch,
                state.key,
            )
            state.step += 1

            if state.step == initial_step + 1:
                t_window_start = time.monotonic()
                last_log_step = state.step
                supervised_tokens_in_window = 0
                if settings.profile_steps > 0:
                    profile_dir = str(run_dir / "profile")
                    jax.profiler.start_trace(profile_dir)
                    logger.info("profile_started", path=profile_dir)

            if (
                settings.profile_steps > 0
                and state.step == initial_step + 1 + settings.profile_steps
            ):
                jax.profiler.stop_trace()
                logger.info(
                    "profile_stopped",
                    path=str(run_dir / "profile"),
                    steps=settings.profile_steps,
                )

            if state.step % settings.log_every == 0 and state.step > last_log_step:
                elapsed = time.monotonic() - t_window_start
                steps_in_window = state.step - last_log_step
                tok_per_s = int(
                    steps_in_window
                    * settings.nominal_tokens_per_step
                    / max(elapsed, 1e-9)
                )
                host_metrics = _collect_host_metrics(
                    step_metrics,
                    lr_schedule,
                    step=state.step,
                    tok_per_s=tok_per_s,
                    steps_in_window=steps_in_window,
                    nominal_tokens_per_step=settings.nominal_tokens_per_step,
                    max_steps=settings.max_steps,
                    supervised_tokens_in_window=supervised_tokens_in_window,
                    elapsed=elapsed,
                )
                reporter.log(step=state.step, metrics=host_metrics.to_dict())
                t_window_start = time.monotonic()
                last_log_step = state.step
                supervised_tokens_in_window = 0

            if state.step % settings.save_every == 0:
                _save()

            if stop_request.requested:
                break

    _save_final_checkpoint_if_needed(state, initial_step=initial_step, save=_save)
    _log_graceful_stop(stop_request, step=state.step, run_dir=run_dir)
