"""MDLM pretraining driver: JIT train step, state loader, ``pretrain`` entry."""

import dataclasses
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import structlog
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from nanodiffusion.checkpoint import (
    load_checkpoint,
    resolve_model_config_from_checkpoint,
    write_config,
)
from nanodiffusion.config import Config
from nanodiffusion.data.cursors import PretrainCursor
from nanodiffusion.data.datasets import get_dataset
from nanodiffusion.data.loader import BatchOutput, pretrain_loader
from nanodiffusion.data.source import TextSource
from nanodiffusion.loop import (
    LoopHyperparams,
    LoopState,
    StepStats,
    resolve_run_dir,
    run_training_loop,
)
from nanodiffusion.loss import TimeSampler, low_discrepancy_sampler
from nanodiffusion.model import (
    DiffusionModel,
    Transformer,
    transformer_skeleton,
)
from nanodiffusion.optimizer import make_optimizer, scale_ema_decay
from nanodiffusion.pretrain.loss import compute_loss
from nanodiffusion.reporter import Reporter, default_sinks
from nanodiffusion.runtime import configure_jax_runtime, place_training_state
from nanodiffusion.schedule import LogLinearSchedule, NoiseSchedule
from nanodiffusion.sharding import setup_mesh
from nanodiffusion.tokenizer import Tokenizer
from nanodiffusion.train_step import TrainStepFn
from nanodiffusion.train_step import make_train_step as make_shared_train_step
from nanodiffusion.types import PRNGKeyArray, Scalar, TokenBatch

logger = structlog.get_logger(__name__)


def make_train_step[M: DiffusionModel](
    optimizer: optax.GradientTransformation,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    ema_decay: float,
    sampler: TimeSampler = low_discrepancy_sampler,
) -> TrainStepFn[M, TokenBatch]:
    """Build an MDLM diffusion train step by binding the pretrain loss."""

    def loss_fn(m: M, batch: TokenBatch, key: PRNGKeyArray) -> Scalar:
        return compute_loss(
            m,
            batch,
            schedule=schedule,
            mask_token_id=mask_token_id,
            key=key,
            sampler=sampler,
        )

    return make_shared_train_step(
        optimizer,
        loss_fn=loss_fn,
        ema_decay=ema_decay,
    )


def _init_source(config: Config) -> TextSource:
    data = config.require_data()
    factory = get_dataset(data.dataset)
    return factory(
        data.data_dir,
        num_train=data.num_train_shards,
        download=False,
    )


def _prepare_batch(
    batch: BatchOutput,
    mesh: Mesh,
) -> tuple[TokenBatch, PretrainCursor, StepStats]:
    """Host → device transfer with async sharding.

    Uses ``jax.device_put(numpy, NamedSharding)`` directly instead of
    ``jnp.asarray`` + ``shard_batch``. ``device_put`` is always async
    so the transfer can overlap with the previous step's tail compute.
    """
    from nanodiffusion.sharding import DP_AXES  # noqa: PLC0415

    sharding = NamedSharding(mesh, P(DP_AXES, None))
    tokens = jax.device_put(batch.tokens, sharding)
    return tokens, batch.state, StepStats()


@dataclasses.dataclass(frozen=True, slots=True)
class _PretrainInitialState:
    """Everything :func:`pretrain`'s loop needs at step 0.

    ``key`` is the loop's RNG: on a fresh run it's the post-init split
    of the seed key; on resume it's restored from the checkpoint so the
    diffusion masking / timestep sampling chain continues from where it
    left off rather than rewinding to step 0.
    """

    model: Transformer
    ema_model: Transformer
    opt_state: optax.OptState
    key: PRNGKeyArray
    step: int
    cursor: PretrainCursor | None


def _load_pretrain_initial_state(
    config: Config,
    optimizer: optax.GradientTransformation,
    *,
    resume_from: Path | None,
    model_key: PRNGKeyArray,
    loop_key: PRNGKeyArray,
) -> _PretrainInitialState:
    """Build the step-0 state for a fresh run or resume from a checkpoint."""
    if resume_from is not None:
        return _load_resumed_pretrain_state(config, optimizer, resume_from)
    return _load_fresh_pretrain_state(config, optimizer, model_key, loop_key)


def _load_fresh_pretrain_state(
    config: Config,
    optimizer: optax.GradientTransformation,
    model_key: PRNGKeyArray,
    loop_key: PRNGKeyArray,
) -> _PretrainInitialState:
    model = Transformer(config.model, key=model_key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    # Deep-copy the model buffers so the EMA starts as an independent
    # replica. Without this, ``model`` and ``ema_model`` alias the same
    # JAX buffers and the ``donate="all"`` train step errors out on the
    # first call ("donate the same buffer twice").
    ema_model = jax.tree.map(lambda x: jnp.copy(x) if eqx.is_array(x) else x, model)
    return _PretrainInitialState(
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        key=loop_key,
        step=0,
        cursor=None,
    )


def _load_resumed_pretrain_state(
    config: Config,
    optimizer: optax.GradientTransformation,
    checkpoint: Path,
) -> _PretrainInitialState:
    model_config = resolve_model_config_from_checkpoint(
        checkpoint,
        fallback=config.model,
        log_event="pretrain_model_config_override",
    )
    skeleton = transformer_skeleton(model_config)

    def build_opt_state(m: Transformer) -> optax.OptState:
        return optimizer.init(eqx.filter(m, eqx.is_inexact_array))

    model, ema_model, opt_state, key, meta = load_checkpoint(
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
        key=key,
        step=meta.step,
        cursor=cursor,
    )


def pretrain(
    config: Config,
    *,
    resume_from: Path | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    profile_steps: int = 0,
) -> Path:
    """Run an MDLM pretraining job end-to-end.

    Returns the run directory so callers (tests, notebooks) can inspect
    the produced artifacts without re-deriving the auto-generated path.
    """
    data = config.require_data()
    train = config.train
    seed_key = jax.random.PRNGKey(train.seed)
    model_key, loop_key = jax.random.split(seed_key)

    optimizer, lr_schedule = make_optimizer(train)
    start = _load_pretrain_initial_state(
        config,
        optimizer,
        resume_from=resume_from,
        model_key=model_key,
        loop_key=loop_key,
    )

    tok = Tokenizer(encode_threads=data.tokenizer_threads)
    source = _init_source(config)
    run_dir = resolve_run_dir(train.run_dir, resume_from=resume_from)
    configure_jax_runtime(run_dir)
    write_config(run_dir, config)
    logger.info(
        "pretrain_start",
        run_dir=str(run_dir),
        dataset=data.dataset,
        max_steps=train.max_steps,
        batch_size=train.batch_size,
        seq_len=config.model.max_seq_len,
        starting_step=start.step,
        resumed=resume_from is not None,
        wandb=wandb_project,
    )

    mesh = setup_mesh()
    ema_decay = scale_ema_decay(train.ema_decay, jax.device_count())
    train_step = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=tok.mask_token_id,
        ema_decay=ema_decay,
    )
    base_loader = pretrain_loader(
        source,
        tok,
        batch_size=train.batch_size,
        seq_len=config.model.max_seq_len,
        split="train",
        tokenizer_batch_size=data.tokenizer_batch_size,
        resume_state=start.cursor,
        max_empty_passes=data.max_empty_passes,
    )

    placed_model, placed_ema_model, placed_opt_state, placed_key = place_training_state(
        start.model,
        start.ema_model,
        start.opt_state,
        start.key,
        mesh,
    )
    state: LoopState[Transformer, PretrainCursor] = LoopState(
        model=placed_model,
        ema_model=placed_ema_model,
        opt_state=placed_opt_state,
        key=placed_key,
        step=start.step,
        cursor=start.cursor,
    )
    settings = LoopHyperparams(
        max_steps=train.max_steps,
        log_every=train.log_every,
        save_every=train.save_every,
        prefetch_size=data.prefetch_size,
        nominal_tokens_per_step=train.batch_size * config.model.max_seq_len,
        event_name="train",
        profile_steps=profile_steps,
    )
    sink_factories = default_sinks(
        event_name="train",
        run_dir=run_dir,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_config=config.model_dump(mode="json"),
    )
    with Reporter(sink_factories) as reporter:
        run_training_loop(
            state,
            run_dir=run_dir,
            train_step=train_step,
            lr_schedule=lr_schedule,
            base_loader=base_loader,
            settings=settings,
            prepare_batch=partial(_prepare_batch, mesh=mesh),
            reporter=reporter,
        )

    logger.info("pretrain_done", step=state.step, run_dir=str(run_dir))
    return run_dir
