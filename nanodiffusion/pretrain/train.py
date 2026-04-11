"""MDLM pretraining driver: JIT train step, state loader, ``pretrain`` entry."""

import dataclasses
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import structlog
from jax import shard_map
from jax.sharding import Mesh
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
    StepMetrics,
    StepStats,
    TrainStepFn,
    resolve_run_dir,
    run_training_loop,
)
from nanodiffusion.loss import TimeSampler, low_discrepancy_sampler
from nanodiffusion.model import (
    DiffusionModel,
    Transformer,
    transformer_skeleton,
)
from nanodiffusion.optimizer import apply_or_skip, make_optimizer, scale_ema_decay
from nanodiffusion.pretrain.loss import compute_loss
from nanodiffusion.reporter import (
    JsonlSink,
    Reporter,
    SinkFactory,
    StructlogSink,
    WandbSink,
)
from nanodiffusion.schedule import LogLinearSchedule, NoiseSchedule
from nanodiffusion.sharding import DP_AXES, replicate, setup_mesh, shard_batch
from nanodiffusion.tokenizer import Tokenizer
from nanodiffusion.types import PRNGKeyArray, Scalar, TokenBatch

logger = structlog.get_logger(__name__)


def make_train_step[M: DiffusionModel](
    optimizer: optax.GradientTransformation,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    ema_decay: float,
    mesh: Mesh | None = None,
    sampler: TimeSampler = low_discrepancy_sampler,
) -> TrainStepFn[M, TokenBatch]:
    """Build an ``eqx.filter_jit`` train step for MDLM diffusion.

    Closures pin ``optimizer``, ``schedule``, ``mask_token_id``,
    ``ema_decay``, and ``sampler`` at trace time so the JIT cache key
    stays stable. The callable is generic over ``M`` so the concrete
    model subclass flows through to the returned tuple without a
    downcast.

    ``mesh`` is the data-parallel mesh; when set, the loss computation
    is wrapped in a :func:`jax.shard_map.shard_map` manual region so
    the Pallas flash-attention kernel inside
    :func:`nanodiffusion.ops.attention` runs on per-device local
    tensors without GSPMD trying (and failing) to partition it.
    Single-device runs can leave it ``None`` — the Pallas call then
    runs in the usual JIT region.

    ``sampler`` defaults to the stratified low-discrepancy sampler used
    by real pretrain runs; examples that need a bounded or biased time
    sampler (e.g. the tiny-model overfit script) can pass their own.

    The returned step emits a metrics dict with ``loss``, ``grad_norm``,
    ``param_norm``, and ``grad_finite``. ``grad_norm`` is the pre-clip
    global norm (so the logged value reflects what the optimizer
    actually saw before clipping kicked in), ``param_norm`` tracks
    post-update weight magnitude as a coarse divergence signal, and
    ``grad_finite`` is 0.0 whenever the step was skipped because the
    gradient norm or loss was non-finite.
    """

    def _compute_loss(m: M, batch: TokenBatch, key: PRNGKeyArray) -> Scalar:
        return compute_loss(
            m,
            batch,
            schedule=schedule,
            mask_token_id=mask_token_id,
            key=key,
            sampler=sampler,
        )

    if mesh is not None:

        def _sharded_loss(m: M, batch: TokenBatch, key: PRNGKeyArray) -> Scalar:
            def _per_shard(m: M, batch: TokenBatch, key: PRNGKeyArray) -> Scalar:
                # Fold the axis indices into the key so every device
                # draws independent diffusion noise; a replicated key
                # would collapse DP back to a single-device run.
                local_key = key
                for axis in DP_AXES:
                    local_key = jax.random.fold_in(local_key, jax.lax.axis_index(axis))
                loss = _compute_loss(m, batch, local_key)
                return jax.lax.pmean(loss, axis_name=DP_AXES)

            return shard_map(  # pyright: ignore[reportCallIssue]
                _per_shard,
                mesh=mesh,
                in_specs=(P(), P(DP_AXES), P()),
                out_specs=P(),
                check_vma=False,
            )(m, batch, key)

        forward = _sharded_loss
    else:
        forward = _compute_loss

    @eqx.filter_jit(donate="all")
    def train_step(
        model: M,
        ema_model: M,
        opt_state: optax.OptState,
        batch: TokenBatch,
        key: PRNGKeyArray,
    ) -> tuple[M, M, optax.OptState, StepMetrics]:
        def loss_fn(m: M) -> Scalar:
            return forward(m, batch, key)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        grad_norm = optax.tree.norm(grads)
        finite = jnp.isfinite(grad_norm) & jnp.isfinite(loss)
        new_model, new_ema_model, new_opt_state = apply_or_skip(
            finite,
            optimizer=optimizer,
            model=model,
            ema_model=ema_model,
            opt_state=opt_state,
            grads=grads,
            ema_decay=ema_decay,
        )
        param_norm = optax.tree.norm(eqx.filter(new_model, eqx.is_inexact_array))
        metrics: StepMetrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "grad_finite": finite.astype(jnp.float32),
        }
        return new_model, new_ema_model, new_opt_state, metrics

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
    mesh: Mesh,
) -> tuple[TokenBatch, PretrainCursor, StepStats]:
    """Host → JAX conversion + device sharding for pretrain batches."""
    return shard_batch(jnp.asarray(batch.tokens), mesh), batch.state, StepStats()


@dataclasses.dataclass(frozen=True, slots=True)
class _PretrainInitialState:
    """Everything :func:`pretrain`'s loop needs at step 0."""

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
    """Build the step-0 state for a fresh run or resume from a checkpoint."""
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
    # Deep-copy the model buffers so the EMA starts as an independent
    # replica. Without this, ``model`` and ``ema_model`` alias the same
    # JAX buffers and the ``donate="all"`` train step errors out on the
    # first call ("donate the same buffer twice").
    ema_model = jax.tree.map(lambda x: jnp.copy(x) if eqx.is_array(x) else x, model)
    return _PretrainInitialState(
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
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


def _default_pretrain_sinks(
    run_dir: Path,
    *,
    config: Config,
    wandb_project: str | None,
    wandb_entity: str | None,
) -> list[SinkFactory]:
    """Assemble the default sink stack for a pretrain run.

    Always emits structlog (so console output stays identical to the
    pre-reporter world) plus a per-run JSONL file under ``run_dir`` for
    offline analysis. Wandb is optional and only enabled when a
    project is supplied; when enabled it runs in the reporter's worker
    process and never imports inside the training process.
    """
    factories: list[SinkFactory] = [
        partial(StructlogSink, "train"),
        partial(JsonlSink, run_dir / "metrics.jsonl"),
    ]
    if wandb_project is not None:
        factories.append(
            partial(
                WandbSink,
                project=wandb_project,
                entity=wandb_entity,
                run_name=run_dir.name,
                config=config.model_dump(mode="json"),
            )
        )
    return factories


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
    jax.config.update("jax_compilation_cache_dir", str(run_dir / ".jax_cache"))
    jax.config.update("jax_explain_cache_misses", True)  # noqa: FBT003
    jax.config.update("jax_optimization_level", "O1")
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
        wandb=wandb_project,
    )

    mesh = setup_mesh()
    ema_decay = scale_ema_decay(config.train.ema_decay, jax.device_count())
    train_step = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=tok.mask_token_id,
        ema_decay=ema_decay,
        mesh=mesh,
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
        model=replicate(start.model, mesh),
        ema_model=replicate(start.ema_model, mesh),
        opt_state=replicate(start.opt_state, mesh),
        key=replicate(key, mesh),
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
        profile_steps=profile_steps,
    )
    sink_factories = _default_pretrain_sinks(
        run_dir,
        config=config,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    with Reporter(sink_factories) as reporter:
        run_training_loop(
            state,
            config=config,
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
