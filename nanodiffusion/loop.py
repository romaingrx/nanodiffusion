"""Shared training loop driving pretrain and SFT.

Paradigm-specific differences (batch pytree shape, optional per-step
metrics) are threaded through a :class:`PrepareBatch` callback and the
``StepMetrics`` dict returned by :data:`TrainStepFn`, so
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
import jax.numpy as jnp
import optax
import structlog
from pydantic import BaseModel

from nanodiffusion.checkpoint import save_checkpoint, write_config
from nanodiffusion.data.cursors import LoaderCursor
from nanodiffusion.data.loader import prefetch
from nanodiffusion.model import DiffusionModel
from nanodiffusion.reporter import MetricReporter
from nanodiffusion.types import PRNGKeyArray, Scalar

logger = structlog.get_logger(__name__)


type StepMetrics = dict[str, Scalar]
"""Per-step metrics returned from a JIT'd train step.

Keys are paradigm-specific (pretrain vs SFT decide what to expose), but
the loop requires the ``"loss"`` key to be present so it can run the
divergence (non-finite) guard at each log boundary.
"""


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


type TrainStepFn[M: DiffusionModel, JB] = Callable[
    [M, M, optax.OptState, JB, PRNGKeyArray],
    tuple[M, M, optax.OptState, StepMetrics],
]

type PrepareBatch[B, JB, C: LoaderCursor] = Callable[[B], tuple[JB, C, StepStats]]


def _collect_host_metrics(
    step_metrics: StepMetrics,
    lr_schedule: optax.Schedule,
    step: int,
    tok_per_s: int,
    supervised_tokens_in_window: int,
    elapsed: float,
) -> dict[str, float | int | str]:
    """Merge device-side step metrics with host-side throughput + HBM.

    Non-finite losses no longer raise — the JIT'd train step now skips
    the model update on non-finite gradients (see
    :func:`nanodiffusion.optimizer.apply_or_skip`), so training can
    recover from transient spikes. We still surface a warning so the
    operator sees them and can correlate with ``grad_finite=0`` in the
    metrics stream.
    """
    host: dict[str, float | int | str] = {k: float(v) for k, v in step_metrics.items()}
    loss_value = host.get("loss")
    if loss_value is None or not math.isfinite(float(loss_value)):
        logger.warning(
            "non_finite_loss",
            step=step,
            loss=loss_value,
            grad_finite=host.get("grad_finite"),
        )
    host["lr"] = jnp.asarray(lr_schedule(step)).item()
    host["tok_per_s"] = tok_per_s
    if supervised_tokens_in_window > 0:
        host["supervised_tok_per_s"] = int(
            supervised_tokens_in_window / max(elapsed, 1e-9)
        )
    mem = jax.devices()[0].memory_stats()
    if mem is not None:
        host["hbm_used_gb"] = round(mem["bytes_in_use"] / 1e9, 2)
        host["hbm_peak_gb"] = round(mem["peak_bytes_in_use"] / 1e9, 2)
    host["num_devices"] = jax.device_count()
    return host


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

    Non-finite losses surface as a :class:`RuntimeError` at the next
    log boundary so a divergent run fails fast instead of burning
    budget on garbage updates.

    Device placement (replicate / shard) is the caller's responsibility:
    the ``state`` pytrees should already be placed on the target devices,
    and ``prepare_batch`` should shard the batch before returning it.
    This keeps the loop device-agnostic and avoids importing the
    sharding module here.
    """
    initial_step = state.step
    last_log_step = state.step
    supervised_tokens_in_window = 0
    latest_metrics: StepMetrics = {}

    def _save() -> None:
        ckpt_dir = run_dir / f"step_{state.step}"
        save_checkpoint(
            ckpt_dir,
            model=state.model,
            ema_model=state.ema_model,
            opt_state=state.opt_state,
            step=state.step,
            cursor=state.cursor,
            update_latest=True,
        )
        write_config(ckpt_dir, config)
        state.last_saved_step = state.step
        logger.info("checkpoint_saved", path=str(ckpt_dir), step=state.step)

    t_window_start = time.monotonic()
    with prefetch(base_loader, size=settings.prefetch_size) as loader:
        for raw_batch in loader:
            if state.step >= settings.max_steps:
                break
            jax_batch, cursor, stats = prepare_batch(raw_batch)
            state.cursor = cursor
            supervised_tokens_in_window += stats.supervised_tokens
            state.key, step_key = jax.random.split(state.key)
            state.model, state.ema_model, state.opt_state, latest_metrics = train_step(
                state.model, state.ema_model, state.opt_state, jax_batch, step_key
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
                    latest_metrics,
                    lr_schedule,
                    step=state.step,
                    tok_per_s=tok_per_s,
                    supervised_tokens_in_window=supervised_tokens_in_window,
                    elapsed=elapsed,
                )
                reporter.log(step=state.step, metrics=host_metrics)
                t_window_start = time.monotonic()
                last_log_step = state.step
                supervised_tokens_in_window = 0

            if state.step % settings.save_every == 0:
                _save()

    if state.step > initial_step and state.last_saved_step != state.step:
        _save()
