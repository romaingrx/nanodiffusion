"""MDLM pretraining primitives: optimizer, EMA, JIT train step, loop.

The public surface is the :func:`pretrain` entry point the CLI calls.
Everything below it is factored into small helpers so they can be unit
tested without spinning up the data pipeline: :func:`make_optimizer`,
:func:`ema_update`, and :func:`make_train_step`.
"""

import dataclasses
import datetime
import time
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import structlog
import yaml

from nanodiffusion.checkpoint import load_checkpoint, save_checkpoint
from nanodiffusion.config import Config, TrainConfig
from nanodiffusion.data.datasets import get_dataset
from nanodiffusion.data.loader import prefetch, pretrain_loader
from nanodiffusion.data.source import SourcePosition, TextSource
from nanodiffusion.loss import compute_loss
from nanodiffusion.model import DiffusionModel, Transformer
from nanodiffusion.schedule import LogLinearSchedule, NoiseSchedule
from nanodiffusion.tokenizer import Tokenizer
from nanodiffusion.types import PRNGKeyArray, Scalar, TokenBatch

logger = structlog.get_logger(__name__)


type TrainStepFn[M: DiffusionModel] = Callable[
    [M, M, optax.OptState, TokenBatch, PRNGKeyArray],
    tuple[M, M, optax.OptState, Scalar],
]


def make_optimizer(
    train_cfg: TrainConfig,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Warmup + cosine-decay AdamW with global-norm grad clipping.

    Returns the optimizer and the lr schedule; the schedule is exposed
    separately so callers can log the current learning rate without
    reaching into opt_state internals.
    """
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        decay_steps=train_cfg.max_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.adamw(lr_schedule, weight_decay=train_cfg.weight_decay),
    )
    return optimizer, lr_schedule


def ema_update[M: eqx.Module](ema_model: M, model: M, decay: float) -> M:
    """Polyak EMA on the float leaves only.

    ``ema_new = decay * ema_old + (1 - decay) * model``. Non-inexact
    leaves (ints, static fields, strings) are left untouched so integer
    bookkeeping arrays are never silently cast to float.
    """
    ema_arrays, static = eqx.partition(ema_model, eqx.is_inexact_array)
    model_arrays, _ = eqx.partition(model, eqx.is_inexact_array)
    new_ema_arrays = jax.tree.map(
        lambda e, m: decay * e + (1.0 - decay) * m, ema_arrays, model_arrays
    )
    return eqx.combine(new_ema_arrays, static)


def make_train_step[M: DiffusionModel](
    optimizer: optax.GradientTransformation,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    ema_decay: float,
) -> TrainStepFn[M]:
    """Build an ``eqx.filter_jit`` train step for MDLM diffusion.

    Closing over ``optimizer``, ``schedule``, ``mask_token_id`` and
    ``ema_decay`` pins them at trace time so the JIT cache key stays
    stable across calls. The returned callable is generic over ``M``
    so its first positional argument ties the model, EMA, and returned
    tuple together at the caller's concrete subclass.
    """

    @eqx.filter_jit
    def train_step(
        model: M,
        ema_model: M,
        opt_state: optax.OptState,
        batch: TokenBatch,
        key: PRNGKeyArray,
    ) -> tuple[M, M, optax.OptState, Scalar]:
        def loss_fn(m: M) -> Scalar:
            return compute_loss(
                m,
                batch,
                schedule=schedule,
                mask_token_id=mask_token_id,
                key=key,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        new_model = eqx.apply_updates(model, updates)
        new_ema_model = ema_update(ema_model, new_model, ema_decay)
        return new_model, new_ema_model, new_opt_state, loss

    return train_step


def _make_run_id() -> str:
    return datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")


def _write_config(run_dir: Path, config: Config) -> None:
    (run_dir / "config.yaml").write_text(yaml.dump(config.model_dump(mode="json")))


def _init_run_dir(
    config: Config, *, starting_step: int, resume_from: Path | None
) -> Path:
    """Resolve the run directory for this invocation.

    Fresh runs land under ``config.train.run_dir/<timestamp>``. Resumes
    reuse ``resume_from.parent`` so every artifact of a logical run —
    logs, checkpoints, config — stays in one place, and a user can
    ``tail -f`` across restarts.
    """
    if resume_from is not None:
        run_dir = resume_from.parent.resolve()
    else:
        run_dir = config.train.run_dir / _make_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_config(run_dir, config)
    logger.info(
        "pretrain_start",
        run_dir=str(run_dir),
        dataset=config.data.dataset,
        max_steps=config.train.max_steps,
        batch_size=config.train.batch_size,
        seq_len=config.model.max_seq_len,
        starting_step=starting_step,
        resumed=resume_from is not None,
    )
    return run_dir


def _init_source(config: Config) -> TextSource:
    factory = get_dataset(config.data.dataset)
    return factory(
        config.data.data_dir,
        num_train=config.data.num_train_shards,
        download=False,
    )


@dataclasses.dataclass
class _LoopState[M: DiffusionModel]:
    """Mutable per-iteration state threaded through the training loop.

    Bundled into a dataclass so :func:`_run_loop` keeps a flat signature
    and :func:`pretrain` stays under the 50-statement lint limit. Not a
    pytree — equinox never sees this.
    """

    model: M
    ema_model: M
    opt_state: optax.OptState
    key: PRNGKeyArray
    step: int
    cursor: SourcePosition | None
    last_saved_step: int | None = None


def _run_loop[M: DiffusionModel](
    state: _LoopState[M],
    *,
    config: Config,
    run_dir: Path,
    train_step: TrainStepFn[M],
    lr_schedule: optax.Schedule,
    base_loader: "object",
) -> None:
    """Inner training loop.

    Mutates ``state`` in place (Python semantics; not a jit'd function).
    ``base_loader`` is typed as ``object`` because the data loader's
    iterator type lives in ``nanodiffusion.data.loader`` and importing
    it here creates a cycle; the value is only passed straight through
    to :func:`prefetch`.
    """
    initial_step = state.step
    last_log_step = state.step
    tokens_per_step = config.train.batch_size * config.model.max_seq_len

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
        _write_config(ckpt_dir, config)
        state.last_saved_step = state.step
        logger.info("checkpoint_saved", path=str(ckpt_dir), step=state.step)

    t_window_start = time.monotonic()
    with prefetch(base_loader, size=config.data.prefetch_size) as loader:  # pyright: ignore[reportArgumentType]
        for batch in loader:
            if state.step >= config.train.max_steps:
                break
            state.cursor = batch.state
            tokens = jnp.asarray(batch.tokens)
            state.key, step_key = jax.random.split(state.key)
            state.model, state.ema_model, state.opt_state, loss = train_step(
                state.model, state.ema_model, state.opt_state, tokens, step_key
            )
            state.step += 1

            if state.step == initial_step + 1:
                # First post-compile step: restart throughput window to
                # exclude JIT compile time from the first tok/s report.
                t_window_start = time.monotonic()
                last_log_step = state.step

            if state.step % config.train.log_every == 0 and state.step > last_log_step:
                elapsed = time.monotonic() - t_window_start
                steps_in_window = state.step - last_log_step
                tok_per_s = int(steps_in_window * tokens_per_step / max(elapsed, 1e-9))
                logger.info(
                    "train",
                    step=state.step,
                    loss=float(loss),
                    lr=jnp.asarray(lr_schedule(state.step)).item(),
                    tok_per_s=tok_per_s,
                )
                t_window_start = time.monotonic()
                last_log_step = state.step

            if state.step % config.train.save_every == 0:
                _save()

    # End-of-training save. Skipped if the loop was a no-op (resume at
    # max_steps) or the previous iteration already wrote this step via
    # the periodic branch — ``latest`` still points at the right place.
    if state.step > initial_step and state.last_saved_step != state.step:
        _save()


def pretrain(
    config: Config,
    *,
    resume_from: Path | None = None,
) -> Path:
    """Run an MDLM pretraining job end-to-end.

    Returns the run directory so callers (tests, notebooks) can inspect
    the produced artifacts without re-deriving the auto-generated path.
    """
    key = jax.random.PRNGKey(config.train.seed)
    key, model_key = jax.random.split(key)

    model = Transformer(config.model, key=model_key)
    ema_model = model
    optimizer, lr_schedule = make_optimizer(config.train)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    step = 0
    cursor: SourcePosition | None = None
    if resume_from is not None:
        model, ema_model, opt_state, meta = load_checkpoint(
            resume_from, model_skeleton=model, opt_state_skeleton=opt_state
        )
        step, cursor = meta.step, meta.cursor
        logger.info("resumed_checkpoint", path=str(resume_from), step=step)

    tok = Tokenizer(encode_threads=config.data.tokenizer_threads)
    source = _init_source(config)
    run_dir = _init_run_dir(config, starting_step=step, resume_from=resume_from)

    train_step = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=tok.mask_token_id,
        ema_decay=config.train.ema_decay,
    )
    base_loader = pretrain_loader(
        source,
        tok,
        batch_size=config.train.batch_size,
        seq_len=config.model.max_seq_len,
        split="train",
        tokenizer_batch_size=config.data.tokenizer_batch_size,
        resume_state=cursor,
        max_empty_passes=config.data.max_empty_passes,
    )

    state = _LoopState(
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        key=key,
        step=step,
        cursor=cursor,
    )
    _run_loop(
        state,
        config=config,
        run_dir=run_dir,
        train_step=train_step,
        lr_schedule=lr_schedule,
        base_loader=base_loader,
    )

    logger.info("pretrain_done", step=state.step, run_dir=str(run_dir))
    return run_dir
