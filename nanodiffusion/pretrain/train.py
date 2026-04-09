"""MDLM pretraining primitives: optimizer, EMA, JIT train step, loop.

The public surface is the :func:`pretrain` entry point the CLI calls.
Everything below it is factored into small helpers so they can be unit
tested without spinning up the data pipeline: :func:`make_optimizer`,
:func:`ema_update`, and :func:`make_train_step`.
"""

import dataclasses
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import structlog

from nanodiffusion.checkpoint import load_checkpoint
from nanodiffusion.config import Config, OptimizerHyperparams
from nanodiffusion.data.cursors import PretrainCursor
from nanodiffusion.data.datasets import get_dataset
from nanodiffusion.data.loader import BatchOutput, pretrain_loader
from nanodiffusion.data.source import TextSource
from nanodiffusion.loop import (
    LoopHyperparams,
    LoopState,
    StepStats,
    TrainStepFn,
    resolve_model_config_from_checkpoint,
    resolve_run_dir,
    run_training_loop,
    write_config,
)
from nanodiffusion.model import (
    DiffusionModel,
    Transformer,
    transformer_skeleton,
)
from nanodiffusion.pretrain.loss import compute_loss
from nanodiffusion.schedule import LogLinearSchedule, NoiseSchedule
from nanodiffusion.tokenizer import Tokenizer
from nanodiffusion.types import PRNGKeyArray, Scalar, TokenBatch

logger = structlog.get_logger(__name__)


def make_optimizer(
    hp: OptimizerHyperparams,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Warmup + cosine-decay AdamW with global-norm grad clipping.

    Typed against :class:`OptimizerHyperparams` so both ``TrainConfig``
    (pretrain) and ``SFTConfig`` (fine-tuning) satisfy it structurally
    — this keeps SFT from dragging in a pretrain config reference just
    to reuse the optimizer factory. Returns the optimizer and the lr
    schedule; the schedule is exposed separately so callers can log
    the current learning rate without reaching into opt_state internals.
    """
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=hp.learning_rate,
        warmup_steps=hp.warmup_steps,
        decay_steps=hp.max_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(hp.grad_clip),
        optax.adamw(lr_schedule, weight_decay=hp.weight_decay),
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
) -> TrainStepFn[M, TokenBatch]:
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


def _init_source(config: Config) -> TextSource:
    factory = get_dataset(config.data.dataset)
    return factory(
        config.data.data_dir,
        num_train=config.data.num_train_shards,
        download=False,
    )


def _prepare_batch(
    batch: BatchOutput,
) -> tuple[TokenBatch, PretrainCursor, StepStats]:
    """Host → JAX conversion for pretrain batches.

    Pretrain doesn't track supervised tokens separately — every token
    contributes to the unconditional diffusion loss — so the stats
    payload is empty and the shared loop omits the
    ``supervised_tok_per_s`` field from its log output.
    """
    return jnp.asarray(batch.tokens), batch.state, StepStats()


@dataclasses.dataclass(frozen=True, slots=True)
class _PretrainInitialState:
    """Everything :func:`pretrain`'s loop needs at step 0.

    Mirrors :class:`nanodiffusion.sft.train._SFTInitialState`; keeping
    the shape identical lets the two drivers be read side-by-side with
    no structural differences beyond the paradigm-specific cursor type.
    """

    model: Transformer
    ema_model: Transformer
    opt_state: optax.OptState
    step: int
    cursor: PretrainCursor | None


def _load_pretrain_initial_state(
    config: Config,
    optimizer: optax.GradientTransformation,
    *,
    resume_from: Path | None,
    model_key: PRNGKeyArray,
) -> _PretrainInitialState:
    """Build the step-0 state for a fresh run or resume from a checkpoint.

    Resume uses :func:`transformer_skeleton` so the abstract shape
    tree feeds :func:`load_checkpoint` without allocating real
    parameter tensors first — the two-phase load is hidden inside
    :func:`load_checkpoint` via the ``opt_state_builder`` callback.
    """
    if resume_from is not None:
        return _load_resumed_pretrain_state(config, optimizer, resume_from)
    return _load_fresh_pretrain_state(config, optimizer, model_key)


def _load_fresh_pretrain_state(
    config: Config,
    optimizer: optax.GradientTransformation,
    model_key: PRNGKeyArray,
) -> _PretrainInitialState:
    model = Transformer(config.model, key=model_key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    return _PretrainInitialState(
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=0,
        cursor=None,
    )


def _load_resumed_pretrain_state(
    config: Config,
    optimizer: optax.GradientTransformation,
    checkpoint: Path,
) -> _PretrainInitialState:
    # ``transformer_skeleton`` builds a zero-cost shape tree so we
    # avoid a full real-weights init on resume. ``load_checkpoint``
    # internally does the two-phase load via ``opt_state_builder`` —
    # see its docstring.
    model_config = resolve_model_config_from_checkpoint(
        checkpoint,
        fallback=config.model,
        log_event="pretrain_model_config_override",
    )
    skeleton = transformer_skeleton(model_config)

    def build_opt_state(m: Transformer) -> optax.OptState:
        return optimizer.init(eqx.filter(m, eqx.is_inexact_array))

    model, ema_model, opt_state, meta = load_checkpoint(
        checkpoint,
        model_skeleton=skeleton,
        opt_state_builder=build_opt_state,
    )
    cursor = meta.require_cursor(PretrainCursor)
    logger.info("resumed_checkpoint", path=str(checkpoint), step=meta.step)
    return _PretrainInitialState(
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        step=meta.step,
        cursor=cursor,
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

    optimizer, lr_schedule = make_optimizer(config.train)
    start = _load_pretrain_initial_state(
        config,
        optimizer,
        resume_from=resume_from,
        model_key=model_key,
    )

    tok = Tokenizer(encode_threads=config.data.tokenizer_threads)
    source = _init_source(config)
    run_dir = resolve_run_dir(config.train.run_dir, resume_from=resume_from)
    write_config(run_dir, config)
    logger.info(
        "pretrain_start",
        run_dir=str(run_dir),
        dataset=config.data.dataset,
        max_steps=config.train.max_steps,
        batch_size=config.train.batch_size,
        seq_len=config.model.max_seq_len,
        starting_step=start.step,
        resumed=resume_from is not None,
    )

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
        resume_state=start.cursor,
        max_empty_passes=config.data.max_empty_passes,
    )

    state: LoopState[Transformer, PretrainCursor] = LoopState(
        model=start.model,
        ema_model=start.ema_model,
        opt_state=start.opt_state,
        key=key,
        step=start.step,
        cursor=start.cursor,
    )
    settings = LoopHyperparams(
        max_steps=config.train.max_steps,
        log_every=config.train.log_every,
        save_every=config.train.save_every,
        prefetch_size=config.data.prefetch_size,
        nominal_tokens_per_step=config.train.batch_size * config.model.max_seq_len,
        event_name="train",
    )
    run_training_loop(
        state,
        config=config,
        run_dir=run_dir,
        train_step=train_step,
        lr_schedule=lr_schedule,
        base_loader=base_loader,
        settings=settings,
        prepare_batch=_prepare_batch,
    )

    logger.info("pretrain_done", step=state.step, run_dir=str(run_dir))
    return run_dir
