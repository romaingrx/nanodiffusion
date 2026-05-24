"""On-disk checkpoints via Orbax + TensorStore."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import structlog
import yaml
from etils import epath
from pydantic import BaseModel, ConfigDict, Field

from nanodiffusion.config import Config, ModelConfig
from nanodiffusion.constants import CONFIG_SIDECAR_FILENAME
from nanodiffusion.data.cursors import LoaderCursor
from nanodiffusion.model import DiffusionModel

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

    import optax

    from nanodiffusion.types import PRNGKeyArray

logger = structlog.get_logger(__name__)

type ModelSnapshot = Literal["model", "ema"]


class CheckpointMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0)
    cursor: LoaderCursor | None = None

    def require_cursor[C: LoaderCursor](self, kind: type[C]) -> C | None:
        if self.cursor is None:
            return None
        if not isinstance(self.cursor, kind):
            msg = (
                f"Checkpoint cursor kind {self.cursor.kind!r} does not "
                f"match expected kind {kind.__name__!r}"
            )
            raise TypeError(msg)
        return self.cursor


def resolve_checkpoint_uri(local_run_dir: Path, *, bucket: str | None) -> str:
    """Return ``gs://<bucket>/<local_run_dir>`` or the local path when no bucket.

    ``local_run_dir`` must be a path under ``cwd`` (typical for training runs:
    ``runs/<paradigm>/<id>``). The local view stays the gcsfuse mount for
    ``.jax_cache``, ``metrics.jsonl``, ``profile/``, ``config.yaml``; Orbax
    bypasses the mount and writes ``step_*/`` directly through TensorStore.
    """
    if bucket is None:
        return str(local_run_dir)
    rel = local_run_dir.resolve().relative_to(Path.cwd())
    return f"gs://{bucket}/{rel.as_posix()}"


def make_manager(
    uri: str | os.PathLike[str], *, max_to_keep: int = 5
) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        epath.Path(uri),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            step_prefix="step",
            enable_async_checkpointing=True,
        ),
        item_names=("state", "meta"),
    )


def save_checkpoint[M: DiffusionModel](
    mngr: ocp.CheckpointManager,
    step: int,
    *,
    model: M,
    ema_model: M,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
    cursor: LoaderCursor | None,
) -> None:
    """Submit an async save; must be called on every host (Orbax coordinates)."""
    state: dict[str, object] = {
        "model": model,
        "ema": ema_model,
        "opt": opt_state,
        "key": key,
    }
    # device_get materialises to host memory so the next step's
    # donate="all" can reuse the device buffers while Orbax uploads.
    arrays = jax.device_get(eqx.filter(state, eqx.is_array))
    meta = CheckpointMeta(step=step, cursor=cursor).model_dump(mode="json")
    mngr.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(arrays),
            meta=ocp.args.JsonSave(meta),
        ),
    )


def load_checkpoint[M: DiffusionModel](
    mngr_or_uri: ocp.CheckpointManager | str | os.PathLike[str],
    *,
    model_skeleton: M,
    opt_state_builder: Callable[[M], optax.OptState],
    step: int | None = None,
) -> tuple[M, M, optax.OptState, PRNGKeyArray, CheckpointMeta]:
    mngr = (
        mngr_or_uri
        if isinstance(mngr_or_uri, ocp.CheckpointManager)
        else make_manager(mngr_or_uri)
    )
    resolved_step = step if step is not None else mngr.latest_step()
    if resolved_step is None:
        msg = f"no finalised checkpoint to restore in {mngr.directory}"
        raise FileNotFoundError(msg)

    skel_state = _build_skel_state(model_skeleton, opt_state_builder)
    arr_skel, static = eqx.partition(skel_state, eqx.is_array)

    restored = mngr.restore(
        resolved_step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(arr_skel),
            meta=ocp.args.JsonRestore(),
        ),
    )
    state = eqx.combine(restored["state"], static)
    meta = CheckpointMeta.model_validate(restored["meta"])
    _assert_step_counters_match(state["opt"], meta.step)
    return (
        cast("M", state["model"]),
        cast("M", state["ema"]),
        cast("optax.OptState", state["opt"]),
        cast("PRNGKeyArray", state["key"]),
        meta,
    )


def load_meta(
    uri: str | os.PathLike[str], *, step: int | None = None
) -> CheckpointMeta:
    mngr = make_manager(uri)
    resolved_step = step if step is not None else mngr.latest_step()
    if resolved_step is None:
        msg = f"no finalised checkpoint to read in {uri}"
        raise FileNotFoundError(msg)
    restored = mngr.restore(
        resolved_step,
        args=ocp.args.Composite(meta=ocp.args.JsonRestore()),
    )
    return CheckpointMeta.model_validate(restored["meta"])


def load_model[M: DiffusionModel](
    uri: str | os.PathLike[str],
    *,
    model_skeleton: M,
    which: ModelSnapshot = "ema",
    step: int | None = None,
) -> M:
    mngr = make_manager(uri)
    resolved_step = step if step is not None else mngr.latest_step()
    if resolved_step is None:
        msg = f"no finalised checkpoint to load in {uri}"
        raise FileNotFoundError(msg)

    real_skeleton = _materialize_skeleton(model_skeleton)
    arr_skel, static = eqx.partition(real_skeleton, eqx.is_array)
    target: dict[str, object] = {which: arr_skel}
    # partial_restore=True is required when target covers only a subtree
    # of the saved state; without it Orbax errors on structure mismatch.
    restored = mngr.restore(
        resolved_step,
        args=ocp.args.Composite(
            state=ocp.args.PyTreeRestore(target, partial_restore=True),
        ),
    )
    return cast("M", eqx.combine(restored["state"][which], static))


def flush(mngr: ocp.CheckpointManager) -> None:
    mngr.wait_until_finished()


def write_config(run_dir: Path, config: BaseModel) -> None:
    (run_dir / CONFIG_SIDECAR_FILENAME).write_text(
        yaml.dump(config.model_dump(mode="json"))
    )


def resolve_model_config_from_checkpoint(
    run_dir: Path,
    *,
    fallback: ModelConfig,
    log_event: str,
) -> ModelConfig:
    """Pick up the run's ``config.yaml`` model section when present.

    The sidecar is authoritative for model shape since the on-disk weights
    were produced under it; a mismatched fallback only triggers a warning
    because a genuine shape mismatch would fail at restore one call later.
    """
    sidecar = run_dir / CONFIG_SIDECAR_FILENAME
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


def _materialize_skeleton[T](skeleton: T) -> T:
    """Zero-fill ``ShapeDtypeStruct`` leaves: ``eqx.is_inexact_array`` (used by
    ``optax.GradientTransformation.init``) returns False for abstract leaves,
    so an unmaterialised skeleton would produce an empty opt_state."""
    return jax.tree.map(
        lambda x: (
            jnp.zeros(x.shape, x.dtype) if isinstance(x, jax.ShapeDtypeStruct) else x
        ),
        skeleton,
    )


def _build_skel_state[M: DiffusionModel](
    model_skeleton: M,
    opt_state_builder: Callable[[M], optax.OptState],
) -> dict[str, object]:
    real_skeleton = _materialize_skeleton(model_skeleton)
    return {
        "model": real_skeleton,
        "ema": real_skeleton,
        "opt": opt_state_builder(real_skeleton),
        "key": jax.random.key(0),
    }


def _assert_step_counters_match(opt_state: object, meta_step: int) -> None:
    """Log a warning if optax's internal ``count`` drifts from ``meta.step``.

    Drift signals a save happening on a step where the optimizer was skipped
    (see :func:`nanodiffusion.optimizer.apply_or_skip`); we warn rather than
    fail so a single skip doesn't abort the resume."""

    def has_scalar_count(x: object) -> bool:
        count = getattr(x, "count", None)
        return isinstance(count, jax.Array) and count.shape == ()

    counts = {
        int(leaf.count)
        for leaf in jax.tree.leaves(opt_state, is_leaf=has_scalar_count)
        if has_scalar_count(leaf)
    }
    if not counts:
        return
    if len(counts) > 1:
        logger.warning(
            "checkpoint_step_counter_disagreement",
            counts=sorted(counts),
            meta_step=meta_step,
        )
        return
    actual = next(iter(counts))
    if actual != meta_step:
        logger.warning(
            "checkpoint_step_counter_mismatch",
            opt_state_count=actual,
            meta_step=meta_step,
        )
