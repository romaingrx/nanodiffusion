"""Shared training loop driving pretrain and SFT."""

import dataclasses
import datetime
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import jax
import optax
import structlog

from nanodiffusion.checkpoint import (
    flush,
    make_manager,
    resolve_checkpoint_uri,
    save_checkpoint_with_logging,
)
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
    return datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")


def resolve_run_dir(run_dir_root: Path, *, resume_from: Path | None) -> Path:
    if resume_from is not None:
        run_dir = resume_from.resolve()
    else:
        run_dir = run_dir_root / make_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_lineage(
    run_dir: Path,
    *,
    paradigm: str,
    base_uri: str,
    base_step: int,
) -> None:
    """Drop a ``lineage.json`` sidecar pointing at the parent checkpoint.

    Makes downstream branching mechanical: any derived run (sft from
    pretrain, rlhf from sft, eval from anything) can be walked back to
    its origin by reading this file. Best-effort on git_sha — failures
    leave the field null rather than aborting the run.
    """
    git_bin = shutil.which("git")
    if git_bin is None:
        git_sha = None
    else:
        try:
            git_sha = subprocess.check_output(  # noqa: S603
                [git_bin, "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError:
            git_sha = None

    lineage = {
        "paradigm": paradigm,
        "base_uri": base_uri,
        "base_step": base_step,
        "git_sha": git_sha,
        "started_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }
    (run_dir / "lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")


@dataclasses.dataclass(frozen=True, slots=True)
class StepStats:
    supervised_tokens: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class LoopHyperparams:
    max_steps: int
    log_every: int
    save_every: int
    prefetch_size: int
    nominal_tokens_per_step: int
    event_name: str
    profile_steps: int = 0


@dataclasses.dataclass
class LoopState[M: DiffusionModel, C: LoaderCursor]:
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
    """Submit a final save if training advanced past the last submitted step.

    Orbax's internal queue-depth-1 means any subsequent ``save()`` already
    blocks on the prior, so no manual pre-flush is needed.
    """
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
    run_dir: Path,
    train_step: TrainStepFn[M, JB],
    lr_schedule: optax.Schedule,
    base_loader: Iterator[B],
    settings: LoopHyperparams,
    prepare_batch: PrepareBatch[B, JB, C],
    reporter: MetricReporter,
) -> None:
    """Drive the training loop in place.

    Device placement is the caller's responsibility: ``state`` pytrees
    must already be on-device and ``prepare_batch`` must shard the batch
    before returning it.
    """
    initial_step = state.step
    last_log_step = state.step
    supervised_tokens_in_window = 0

    ckpt_uri = resolve_checkpoint_uri(run_dir, bucket=os.environ.get("GCS_BUCKET"))
    mngr = make_manager(ckpt_uri)
    logger.info("checkpoint_manager_ready", uri=ckpt_uri)

    def _save() -> None:
        save_checkpoint_with_logging(
            mngr,
            state.step,
            model=state.model,
            ema_model=state.ema_model,
            opt_state=state.opt_state,
            key=state.key,
            cursor=state.cursor,
        )
        state.last_saved_step = state.step

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
    flush(mngr)
    _log_graceful_stop(stop_request, step=state.step, run_dir=run_dir)
