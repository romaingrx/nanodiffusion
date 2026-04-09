"""SFT driver: JIT train step, state loader, ``sft_finetune`` entry."""

import dataclasses
from pathlib import Path

import equinox as eqx
import jax
import optax
import structlog

from nanodiffusion.checkpoint import (
    load_checkpoint,
    load_model,
    resolve_model_config_from_checkpoint,
    write_config,
)
from nanodiffusion.config import Config
from nanodiffusion.data.chat_datasets import get_chat_dataset
from nanodiffusion.data.chat_source import ChatSource, TaskMixture
from nanodiffusion.data.cursors import SFTCursor
from nanodiffusion.data.sft_loader import SFTBatchOutput, SFTJaxBatch, sft_loader
from nanodiffusion.loop import (
    LoopHyperparams,
    LoopState,
    StepStats,
    TrainStepFn,
    resolve_run_dir,
    run_training_loop,
)
from nanodiffusion.model import DiffusionModel, Transformer, transformer_skeleton
from nanodiffusion.optimizer import ema_update, make_optimizer
from nanodiffusion.schedule import LogLinearSchedule, NoiseSchedule
from nanodiffusion.sft.loss import compute_sft_loss
from nanodiffusion.tokenizer import Tokenizer
from nanodiffusion.types import PRNGKeyArray, Scalar

logger = structlog.get_logger(__name__)


type SFTTrainStepFn[M: DiffusionModel] = TrainStepFn[M, SFTJaxBatch]


def make_sft_train_step[M: DiffusionModel](
    optimizer: optax.GradientTransformation,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    ema_decay: float,
) -> SFTTrainStepFn[M]:
    """Build an ``eqx.filter_jit`` train step for SFT.

    Unpacks the :class:`SFTJaxBatch` tokens + loss_mask pair that
    :func:`compute_sft_loss` expects. Closures pin ``optimizer``,
    ``schedule``, ``mask_token_id``, and ``ema_decay`` at trace time
    so the JIT cache key stays stable.
    """

    @eqx.filter_jit
    def train_step(
        model: M,
        ema_model: M,
        opt_state: optax.OptState,
        batch: SFTJaxBatch,
        key: PRNGKeyArray,
    ) -> tuple[M, M, optax.OptState, Scalar]:
        def loss_fn(m: M) -> Scalar:
            return compute_sft_loss(
                m,
                batch.tokens,
                batch.loss_mask,
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


def _prepare_sft_batch(
    batch: SFTBatchOutput,
) -> tuple[SFTJaxBatch, SFTCursor, StepStats]:
    """Host → JAX conversion for SFT batches.

    ``supervised_tokens`` is counted on the numpy side *before*
    :meth:`SFTBatchOutput.to_jax` to avoid a JAX→host sync each step.
    """
    supervised = int(batch.loss_mask.sum())
    return batch.to_jax(), batch.state, StepStats(supervised_tokens=supervised)


def _build_task_mixture(config: Config) -> TaskMixture:
    """Resolve ``config.sft.datasets`` into a seeded :class:`TaskMixture`.

    Each ``SFTDatasetConfig(name, epochs)`` contributes its source
    ``epochs`` times via duplicate entries — that's how oversampling
    is expressed without a separate multiplier.
    """
    sources: list[ChatSource] = []
    for entry in config.sft.datasets:
        factory = get_chat_dataset(entry.name)
        src = factory(config.data.data_dir, download=True)
        sources.extend([src] * entry.epochs)
    return TaskMixture(sources, seed=config.sft.seed)


@dataclasses.dataclass(frozen=True, slots=True)
class _SFTInitialState:
    """Everything :func:`sft_finetune`'s loop needs at step 0."""

    model: Transformer
    ema_model: Transformer
    opt_state: optax.OptState
    step: int
    cursor: SFTCursor | None


def _load_sft_initial_state(
    config: Config,
    optimizer: optax.GradientTransformation,
    *,
    pretrain_checkpoint: Path | None,
    resume_from: Path | None,
) -> _SFTInitialState:
    """Validate the starting-point XOR and load weights into one state.

    Exactly one of ``pretrain_checkpoint`` or ``resume_from`` must be
    non-``None``; anything else raises :class:`ValueError`.
    """
    if pretrain_checkpoint is not None and resume_from is not None:
        msg = (
            "sft_finetune requires exactly one of 'pretrain_checkpoint' "
            "or 'resume_from' (got both)"
        )
        raise ValueError(msg)
    if pretrain_checkpoint is not None:
        return _load_fresh_sft_state(config, optimizer, pretrain_checkpoint)
    if resume_from is not None:
        return _load_resumed_sft_state(config, optimizer, resume_from)
    msg = (
        "sft_finetune requires exactly one of 'pretrain_checkpoint' "
        "or 'resume_from' (got neither)"
    )
    raise ValueError(msg)


def _load_fresh_sft_state(
    config: Config,
    optimizer: optax.GradientTransformation,
    checkpoint: Path,
) -> _SFTInitialState:
    model_config = resolve_model_config_from_checkpoint(
        checkpoint,
        fallback=config.model,
        log_event="sft_model_config_override",
    )
    skeleton = transformer_skeleton(model_config)
    model = load_model(checkpoint, model_skeleton=skeleton, which="model")
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    return _SFTInitialState(
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=0,
        cursor=None,
    )


def _load_resumed_sft_state(
    config: Config,
    optimizer: optax.GradientTransformation,
    checkpoint: Path,
) -> _SFTInitialState:
    model_config = resolve_model_config_from_checkpoint(
        checkpoint,
        fallback=config.model,
        log_event="sft_model_config_override",
    )
    skeleton = transformer_skeleton(model_config)

    def build_opt_state(m: Transformer) -> optax.OptState:
        return optimizer.init(eqx.filter(m, eqx.is_inexact_array))

    model, ema_model, opt_state, meta = load_checkpoint(
        checkpoint,
        model_skeleton=skeleton,
        opt_state_builder=build_opt_state,
    )
    cursor = meta.require_cursor(SFTCursor)
    logger.info("sft_resumed", path=str(checkpoint), step=meta.step)
    return _SFTInitialState(
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        step=meta.step,
        cursor=cursor,
    )


def sft_finetune(
    config: Config,
    *,
    pretrain_checkpoint: Path | None = None,
    resume_from: Path | None = None,
) -> Path:
    """Run an SFT fine-tuning job end-to-end.

    Exactly one of ``pretrain_checkpoint`` or ``resume_from`` must be
    provided:

    * ``pretrain_checkpoint`` — fresh SFT run from pretrain weights.
      Loads the raw model (``which="model"``, not EMA, since EMA is a
      sampling-time smoothing that would lag the optimizer's starting
      point) and builds fresh EMA, opt-state, step, and cursor.
    * ``resume_from`` — continue an interrupted SFT run. Loads model,
      EMA, opt-state, and cursor; the loader fast-forwards past the
      saved permutation index so no conversations are re-ingested.
    """
    optimizer, lr_schedule = make_optimizer(config.sft)
    start = _load_sft_initial_state(
        config,
        optimizer,
        pretrain_checkpoint=pretrain_checkpoint,
        resume_from=resume_from,
    )

    tok = Tokenizer(encode_threads=config.sft.tokenizer_threads)
    run_dir = resolve_run_dir(config.sft.run_dir, resume_from=resume_from)
    write_config(run_dir, config)
    logger.info(
        "sft_start",
        run_dir=str(run_dir),
        datasets=[d.name for d in config.sft.datasets],
        max_steps=config.sft.max_steps,
        batch_size=config.sft.batch_size,
        seq_len=config.model.max_seq_len,
        starting_step=start.step,
        resumed=resume_from is not None,
    )

    train_step = make_sft_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=tok.mask_token_id,
        ema_decay=config.sft.ema_decay,
    )

    source = _build_task_mixture(config)
    base_loader = sft_loader(
        source,
        tok,
        batch_size=config.sft.batch_size,
        seq_len=config.model.max_seq_len,
        seed=config.sft.seed,
        resume_state=start.cursor,
        max_empty_passes=config.sft.max_empty_passes,
    )

    state: LoopState[Transformer, SFTCursor] = LoopState(
        model=start.model,
        ema_model=start.ema_model,
        opt_state=start.opt_state,
        key=jax.random.PRNGKey(config.sft.seed),
        step=start.step,
        cursor=start.cursor,
    )
    settings = LoopHyperparams(
        max_steps=config.sft.max_steps,
        log_every=config.sft.log_every,
        save_every=config.sft.save_every,
        prefetch_size=config.sft.prefetch_size,
        nominal_tokens_per_step=config.sft.batch_size * config.model.max_seq_len,
        event_name="sft_train",
    )
    run_training_loop(
        state,
        config=config,
        run_dir=run_dir,
        train_step=train_step,
        lr_schedule=lr_schedule,
        base_loader=base_loader,
        settings=settings,
        prepare_batch=_prepare_sft_batch,
    )

    logger.info("sft_done", step=state.step, run_dir=str(run_dir))
    return run_dir
