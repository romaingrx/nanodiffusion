"""SFT training loop: JIT train step + end-to-end ``sft_finetune`` driver.

Shares the optimizer factory and EMA helper with
:mod:`nanodiffusion.pretrain.train` via structural typing
(:class:`nanodiffusion.config.OptimizerHyperparams`) and the small
run-dir helpers in :mod:`nanodiffusion._loop_utils`. The loop body is
deliberately duplicated from the pretrain path — it is ~40 lines of
mostly-linear state mutation, and sharing it would force a
``BatchAdapter`` indirection that hurts readability more than it helps.
"""

import dataclasses
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import structlog

from nanodiffusion._loop_utils import make_run_id, write_config
from nanodiffusion.checkpoint import load_model, save_checkpoint
from nanodiffusion.config import Config
from nanodiffusion.data.chat_datasets import get_chat_dataset
from nanodiffusion.data.chat_source import ChatSource, TaskMixture
from nanodiffusion.data.loader import prefetch
from nanodiffusion.data.sft_loader import SFTBatchOutput, SFTJaxBatch, sft_loader
from nanodiffusion.data.source import SourcePosition
from nanodiffusion.model import DiffusionModel, Transformer
from nanodiffusion.pretrain.train import ema_update, make_optimizer
from nanodiffusion.schedule import LogLinearSchedule, NoiseSchedule
from nanodiffusion.sft.loss import compute_sft_loss
from nanodiffusion.tokenizer import Tokenizer
from nanodiffusion.types import PRNGKeyArray, Scalar

logger = structlog.get_logger(__name__)


type SFTTrainStepFn[M: DiffusionModel] = Callable[
    [M, M, optax.OptState, SFTJaxBatch, PRNGKeyArray],
    tuple[M, M, optax.OptState, Scalar],
]


def make_sft_train_step[M: DiffusionModel](
    optimizer: optax.GradientTransformation,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    ema_decay: float,
) -> SFTTrainStepFn[M]:
    """Build an ``eqx.filter_jit`` train step for SFT.

    Mirrors :func:`nanodiffusion.pretrain.train.make_train_step` but the
    loss function unpacks an :class:`SFTJaxBatch` into the tokens +
    loss_mask pair that :func:`compute_sft_loss` expects. Closures pin
    ``optimizer``, ``schedule``, ``mask_token_id`` and ``ema_decay`` at
    trace time so the JIT cache key stays stable across calls.
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


def _build_task_mixture(config: Config) -> TaskMixture:
    """Resolve ``config.sft.datasets`` into a seeded :class:`TaskMixture`.

    Each ``SFTDatasetConfig(name, epochs)`` contributes its source
    ``epochs`` times via duplicate entries in the mixture list — the
    same oversampling trick nanochat uses in ``chat_sft.py``.
    """
    sources: list[ChatSource] = []
    for entry in config.sft.datasets:
        factory = get_chat_dataset(entry.name)
        src = factory(config.data.data_dir, download=True)
        sources.extend([src] * entry.epochs)
    return TaskMixture(sources, seed=config.sft.seed)


def _resolve_model_skeleton(checkpoint: Path, config: Config) -> Transformer:
    """Build a Transformer skeleton shape-matching the saved checkpoint.

    Reads ``<checkpoint>/config.yaml`` when available and uses its
    ``model`` section, logging a warning if it differs from the current
    SFT config. Falls back to the current config if the checkpoint
    directory has no sidecar — the resulting shape mismatch will raise
    at deserialization time, which is the right failure mode.
    """
    ckpt_config_path = checkpoint / "config.yaml"
    model_config = config.model
    if ckpt_config_path.exists():
        ckpt_config = Config.from_yaml(ckpt_config_path)
        if ckpt_config.model != config.model:
            logger.warning(
                "sft_model_config_override",
                using=ckpt_config.model.model_dump(),
                ignored=config.model.model_dump(),
            )
        model_config = ckpt_config.model
    key = jax.random.PRNGKey(config.sft.seed)
    return Transformer(model_config, key=key)


@dataclasses.dataclass
class _SFTLoopState[M: DiffusionModel]:
    """Mutable state threaded through the SFT loop body."""

    model: M
    ema_model: M
    opt_state: optax.OptState
    key: PRNGKeyArray
    step: int
    cursor: SourcePosition | None
    last_saved_step: int | None = None


def _init_sft_run_dir(config: Config, *, starting_step: int) -> Path:
    run_dir = config.sft.run_dir / make_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir, config)
    logger.info(
        "sft_start",
        run_dir=str(run_dir),
        datasets=[d.name for d in config.sft.datasets],
        max_steps=config.sft.max_steps,
        batch_size=config.sft.batch_size,
        seq_len=config.model.max_seq_len,
        starting_step=starting_step,
    )
    return run_dir


def _run_sft_loop[M: DiffusionModel](
    state: _SFTLoopState[M],
    *,
    config: Config,
    run_dir: Path,
    train_step: SFTTrainStepFn[M],
    lr_schedule: optax.Schedule,
    base_loader: Iterator[SFTBatchOutput],
) -> None:
    """Inner SFT training loop.

    Mirrors :func:`nanodiffusion.pretrain.train._run_loop`'s shape — the
    duplication is deliberate; see the module docstring for rationale.
    Throughput logging reports both nominal ``tok_per_s`` (fixed-per-step,
    same as pretrain) and ``supervised_tok_per_s`` derived from the
    batch's loss_mask, since SFT's effective learning signal per step is
    a fraction of ``batch_size * seq_len``.
    """
    initial_step = state.step
    last_log_step = state.step
    nominal_tokens_per_step = config.sft.batch_size * config.model.max_seq_len
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
    with prefetch(base_loader, size=config.sft.prefetch_size) as loader:
        for batch_output in loader:
            if state.step >= config.sft.max_steps:
                break
            state.cursor = batch_output.state
            batch = batch_output.to_jax()
            supervised_tokens_in_window += int(batch.loss_mask.sum())
            state.key, step_key = jax.random.split(state.key)
            state.model, state.ema_model, state.opt_state, loss = train_step(
                state.model, state.ema_model, state.opt_state, batch, step_key
            )
            state.step += 1

            if state.step == initial_step + 1:
                # First post-compile step: restart throughput window to
                # exclude JIT compile time from the first tok/s report.
                t_window_start = time.monotonic()
                last_log_step = state.step
                supervised_tokens_in_window = 0

            if state.step % config.sft.log_every == 0 and state.step > last_log_step:
                elapsed = time.monotonic() - t_window_start
                steps_in_window = state.step - last_log_step
                tok_per_s = int(
                    steps_in_window * nominal_tokens_per_step / max(elapsed, 1e-9)
                )
                supervised_tok_per_s = int(
                    supervised_tokens_in_window / max(elapsed, 1e-9)
                )
                logger.info(
                    "train",
                    step=state.step,
                    loss=float(loss),
                    lr=jnp.asarray(lr_schedule(state.step)).item(),
                    tok_per_s=tok_per_s,
                    supervised_tok_per_s=supervised_tok_per_s,
                )
                t_window_start = time.monotonic()
                last_log_step = state.step
                supervised_tokens_in_window = 0

            if state.step % config.sft.save_every == 0:
                _save()

    if state.step > initial_step and state.last_saved_step != state.step:
        _save()


def sft_finetune(
    config: Config,
    *,
    checkpoint: Path,
) -> Path:
    """Run an SFT fine-tuning job end-to-end.

    Starts from the raw trained weights in ``<checkpoint>/model.eqx``
    (``which="model"``, not the EMA — EMA is a sampling-time smoothing
    artifact that would lag the optimizer's starting point). Builds a
    fresh EMA, optimizer state, step counter, and cursor, so SFT is
    always a logically new training run even though the model weights
    are hot from pretrain.
    """
    skeleton = _resolve_model_skeleton(checkpoint, config)
    model = load_model(checkpoint, model_skeleton=skeleton, which="model")
    ema_model = model

    optimizer, lr_schedule = make_optimizer(config.sft)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    tok = Tokenizer(encode_threads=config.sft.tokenizer_threads)
    run_dir = _init_sft_run_dir(config, starting_step=0)

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
        max_empty_passes=config.sft.max_empty_passes,
    )

    state = _SFTLoopState(
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        key=jax.random.PRNGKey(config.sft.seed),
        step=0,
        cursor=None,
    )
    _run_sft_loop(
        state,
        config=config,
        run_dir=run_dir,
        train_step=train_step,
        lr_schedule=lr_schedule,
        base_loader=base_loader,
    )

    logger.info("sft_done", step=state.step, run_dir=str(run_dir))
    return run_dir
