"""Shared training loop driving pretrain and SFT.

Both paradigms run the same three-phase step — pull a host batch from a
prefetched loader, convert it to a JAX batch, run the JIT'd train step —
and share the same throughput-window logging, periodic save, and
first-post-compile timing reset. Only two things genuinely differ:
the batch pytree the train step consumes, and the optional per-step
metrics pretrain doesn't track (e.g. SFT's supervised-token count).
A small :class:`PrepareBatch` callback threads both through, so
:func:`run_training_loop` owns the hairy timing/saving logic and each
paradigm stays focused on its loss and its batch shape.
"""

import dataclasses
import datetime
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import structlog
import yaml
from pydantic import BaseModel

from nanodiffusion.checkpoint import save_checkpoint
from nanodiffusion.config import Config, ModelConfig
from nanodiffusion.data.cursors import LoaderCursor
from nanodiffusion.data.loader import prefetch
from nanodiffusion.model import DiffusionModel
from nanodiffusion.types import PRNGKeyArray, Scalar

logger = structlog.get_logger(__name__)


def make_run_id() -> str:
    """UTC timestamp run id, e.g. ``20260408-193015``."""
    return datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")


def write_config(run_dir: Path, config: BaseModel) -> None:
    """Dump a resolved pydantic config to ``run_dir/config.yaml``.

    Uses ``model_dump(mode="json")`` so ``Path`` and other non-yaml
    types serialize cleanly — matches what :func:`yaml.safe_load` will
    accept back when we reload the config at sampling time.
    """
    (run_dir / "config.yaml").write_text(yaml.dump(config.model_dump(mode="json")))


def resolve_model_config_from_checkpoint(
    checkpoint: Path,
    *,
    fallback: ModelConfig,
    log_event: str,
) -> ModelConfig:
    """Pick up a checkpoint's ``config.yaml`` model section when present.

    The sidecar is the authoritative source for model shape because the
    on-disk weights were produced under it. If the user-supplied
    ``fallback`` disagrees we warn under ``log_event`` and keep going
    under the sidecar — deserialising into a mismatched skeleton would
    fail noisily one line later anyway, which is the right failure
    mode. Missing sidecar falls back silently: hand-constructed
    checkpoints in tests don't need one to work.
    """
    sidecar = checkpoint / "config.yaml"
    if not sidecar.exists():
        return fallback
    from_disk = Config.from_yaml(sidecar)
    if from_disk.model != fallback:
        logger.warning(
            log_event,
            using=from_disk.model.model_dump(),
            ignored=fallback.model_dump(),
        )
    return from_disk.model


def resolve_run_dir(run_dir_root: Path, *, resume_from: Path | None) -> Path:
    """Fresh timestamped run dir, or reuse ``resume_from.parent`` on resume.

    Either way the returned directory exists on disk; callers are
    expected to drop their own ``config.yaml`` / logger.info lines on
    top so that pretrain and SFT can keep their event-name and log-field
    conventions separate.
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

    ``supervised_tokens`` is non-zero only for SFT, where the loader's
    ``loss_mask`` picks out which positions actually contribute
    gradient. Pretrain leaves it at 0 and the loop omits the derived
    ``supervised_tok_per_s`` field from its log output in that case.
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


@dataclasses.dataclass
class LoopState[M: DiffusionModel, C: LoaderCursor]:
    """Mutable per-iteration state threaded through the loop body.

    Not a JAX pytree — equinox never sees this; it's plain Python so
    the loop can update ``step`` / ``cursor`` / ``last_saved_step``
    with normal assignment between jit'd train-step calls.
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
    tuple[M, M, optax.OptState, Scalar],
]

type PrepareBatch[B, JB, C: LoaderCursor] = Callable[[B], tuple[JB, C, StepStats]]


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
) -> None:
    """Drive the training loop in place.

    Mutates ``state`` (model, ema, opt_state, key, step, cursor) via
    Python assignment between JIT'd train-step calls; the function is
    deliberately *not* JIT'd so the prefetch thread, logging, and
    checkpointing can interleave with device work cleanly.

    The throughput window is reset once after the first post-compile
    step so JIT compile latency is excluded from the first ``tok_per_s``
    report, and the end-of-training save is skipped when a periodic
    save already landed on the terminal step.
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
            state.model, state.ema_model, state.opt_state, loss = train_step(
                state.model, state.ema_model, state.opt_state, jax_batch, step_key
            )
            state.step += 1

            if state.step == initial_step + 1:
                t_window_start = time.monotonic()
                last_log_step = state.step
                supervised_tokens_in_window = 0

            if state.step % settings.log_every == 0 and state.step > last_log_step:
                elapsed = time.monotonic() - t_window_start
                steps_in_window = state.step - last_log_step
                tok_per_s = int(
                    steps_in_window
                    * settings.nominal_tokens_per_step
                    / max(elapsed, 1e-9)
                )
                extras: dict[str, int] = {}
                if supervised_tokens_in_window > 0:
                    extras["supervised_tok_per_s"] = int(
                        supervised_tokens_in_window / max(elapsed, 1e-9)
                    )
                logger.info(
                    settings.event_name,
                    step=state.step,
                    loss=float(loss),
                    lr=jnp.asarray(lr_schedule(state.step)).item(),
                    tok_per_s=tok_per_s,
                    **extras,
                )
                t_window_start = time.monotonic()
                last_log_step = state.step
                supervised_tokens_in_window = 0

            if state.step % settings.save_every == 0:
                _save()

    if state.step > initial_step and state.last_saved_step != state.step:
        _save()
