"""MDLM pretraining primitives: optimizer, EMA, JIT train step, loop.

The public surface is the :func:`pretrain` entry point the CLI calls.
Everything below it is factored into small helpers so they can be unit
tested without spinning up the data pipeline: :func:`make_optimizer`,
:func:`ema_update`, and :func:`make_train_step`.
"""

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
    if train_cfg.max_steps <= train_cfg.warmup_steps:
        msg = (
            f"max_steps ({train_cfg.max_steps}) must exceed warmup_steps "
            f"({train_cfg.warmup_steps}); otherwise the cosine schedule has "
            "no decay phase."
        )
        raise ValueError(msg)
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


def _init_run_dir(config: Config, *, starting_step: int) -> Path:
    run_id = _make_run_id()
    run_dir = config.train.run_dir / run_id
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
    )
    return run_dir


def _init_source(config: Config) -> TextSource:
    factory = get_dataset(config.data.dataset)
    return factory(
        config.data.data_dir,
        num_train=config.data.num_train_shards,
        download=False,
    )


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
    run_dir = _init_run_dir(config, starting_step=step)

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

    last_cursor = cursor
    t_window_start = time.monotonic()
    tokens_per_step = config.train.batch_size * config.model.max_seq_len

    def _save(name: str) -> None:
        ckpt_dir = run_dir / name
        save_checkpoint(
            ckpt_dir,
            model=model,
            ema_model=ema_model,
            opt_state=opt_state,
            step=step,
            cursor=last_cursor,
        )
        _write_config(ckpt_dir, config)
        logger.info("checkpoint_saved", path=str(ckpt_dir), step=step)

    with prefetch(base_loader, size=config.data.prefetch_size) as loader:
        for batch in loader:
            if step >= config.train.max_steps:
                break
            last_cursor = batch.state
            jax_batch = batch.to_jax()
            key, step_key = jax.random.split(key)
            model, ema_model, opt_state, loss = train_step(
                model, ema_model, opt_state, jax_batch.tokens, step_key
            )
            step += 1

            if step % config.train.log_every == 0:
                elapsed = time.monotonic() - t_window_start
                tok_per_s = int(
                    config.train.log_every * tokens_per_step / max(elapsed, 1e-9)
                )
                logger.info(
                    "train",
                    step=step,
                    loss=float(loss),
                    lr=jnp.asarray(lr_schedule(step)).item(),
                    tok_per_s=tok_per_s,
                )
                t_window_start = time.monotonic()

            if step % config.train.save_every == 0:
                _save(f"step_{step}")

    _save(f"step_{step}_final")
    logger.info("pretrain_done", step=step, run_dir=str(run_dir))
    return run_dir
