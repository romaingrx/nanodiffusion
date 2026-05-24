"""On-disk checkpoints via Orbax + TensorStore.

A checkpoint manager owns a directory of step subdirectories
(``step_N/``) each holding a ``state/`` item (model + ema + opt_state +
key, serialised through TensorStore's OCDBT format) and a ``meta/``
item (step counter + data-loader cursor, serialised as JSON). The
``config.yaml`` sidecar is written once at run-dir level by
:func:`write_config`, not per-step.

The serialisation switch from ``eqx.tree_serialise_leaves`` was driven
by the gcsfuse mount: Orbax + TensorStore talk the GCS JSON API
directly via chunked OCDBT uploads (100+ MB/s), while the eqx path
goes through gcsfuse and caps at ~3 MB/s because the many small leaf
writes can't be batched.
"""

from __future__ import annotations

import os
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
    from collections.abc import Callable
    from pathlib import Path

    import optax

    from nanodiffusion.types import PRNGKeyArray

logger = structlog.get_logger(__name__)

type ModelSnapshot = Literal["model", "ema"]


class CheckpointMeta(BaseModel):
    """Step counter + data-loader cursor persisted alongside weights.

    Pydantic handles JSON round-trip for the discriminated ``cursor``
    union so save/load stays free of variant checks.
    """

    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0)
    cursor: LoaderCursor | None = None

    def require_cursor[C: LoaderCursor](self, kind: type[C]) -> C | None:
        """Return the cursor narrowed to ``kind`` or raise on mismatch.

        A mismatch means the user pointed ``--resume-from`` at the
        wrong run dir — we fail at the boundary rather than silently
        feeding the wrong cursor type into a loader.
        """
        if self.cursor is None:
            return None
        if not isinstance(self.cursor, kind):
            msg = (
                f"Checkpoint cursor kind {self.cursor.kind!r} does not "
                f"match expected kind {kind.__name__!r}"
            )
            raise TypeError(msg)
        return self.cursor


def resolve_checkpoint_uri(local_run_dir: Path, *, env_var: str = "GCS_BUCKET") -> str:
    """Return the URI Orbax should target for a local run dir.

    When ``$GCS_BUCKET`` is set, maps the ``runs/...`` suffix of
    ``local_run_dir`` onto ``gs://<bucket>/<rel>`` so TensorStore goes
    directly to GCS instead of through gcsfuse. Without the env var
    (local dev, tests), the local path is returned as-is so
    ``epath.Path`` falls back to the regular filesystem.
    """
    bucket = os.environ.get(env_var)
    if bucket is None:
        return str(local_run_dir)
    parts = local_run_dir.absolute().parts
    try:
        idx = parts.index("runs")
    except ValueError:
        return str(local_run_dir)
    rel = "/".join(parts[idx:])
    return f"gs://{bucket}/{rel}"


def make_manager(
    uri: str | os.PathLike[str], *, max_to_keep: int = 5
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager for ``uri``.

    Accepts ``gs://...`` URLs or local paths transparently via
    ``epath.Path``. ``max_to_keep`` triggers GC of older steps after
    each successful save.
    """
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
    """Queue an asynchronous save of ``(model, ema, opt_state, key, meta)``.

    Returns ~immediately. Orbax fans out TensorStore writes on a
    background thread. A subsequent ``save_checkpoint`` call on the
    same manager blocks on the previous save's finalisation (queue
    depth one) — this provides natural backpressure when ``save_every``
    is short relative to upload time.

    Multi-host: must be called on every host. Orbax does its own
    coordination internally; there is no rank-0 gate at the Python
    level. The state tree is materialised to host memory via
    ``jax.device_get`` before submission so the next training step's
    ``donate="all"`` can reuse the device buffers while Orbax is still
    uploading.
    """
    state: dict[str, object] = {
        "model": model,
        "ema": ema_model,
        "opt": opt_state,
        "key": key,
    }
    arrays, _static = eqx.partition(state, eqx.is_array)
    arrays = jax.device_get(arrays)
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
    """Inverse of :func:`save_checkpoint`.

    Accepts either an open ``CheckpointManager`` (when the caller is
    keeping one around for ongoing saves — typical in the training
    loop) or a URI/path (one-shot loads from tests / inference paths).

    ``step`` defaults to ``mngr.latest_step()``. Raises
    :class:`FileNotFoundError` if no finalised step exists.
    """
    mngr = (
        mngr_or_uri
        if isinstance(mngr_or_uri, ocp.CheckpointManager)
        else make_manager(mngr_or_uri, max_to_keep=1)
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
    """Read just the ``meta`` item from a checkpoint.

    Standalone — used by inference paths that don't need the heavy
    model+opt_state restore. Returns the validated Pydantic model
    (step counter + data-loader cursor).
    """
    mngr = make_manager(uri, max_to_keep=1)
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
    """Read just the model (or EMA) weights from a checkpoint.

    Standalone — no manager required from the caller. Opens a manager,
    restores only the requested subtree (Orbax restores the subset of
    leaves provided in the target), and returns. ``which="ema"`` is
    the default for sampling/inference because EMA weights are the
    smoothed inference target; ``which="model"`` returns the raw
    trained weights (used to seed SFT from a pretrain checkpoint).
    """
    mngr = make_manager(uri, max_to_keep=1)
    resolved_step = step if step is not None else mngr.latest_step()
    if resolved_step is None:
        msg = f"no finalised checkpoint to load in {uri}"
        raise FileNotFoundError(msg)

    real_skeleton = _materialize_skeleton(model_skeleton)
    arr_skel, static = eqx.partition(real_skeleton, eqx.is_array)
    target: dict[str, object] = {which: arr_skel}
    # ``partial_restore=True`` tells Orbax that ``target`` intentionally
    # covers only a subtree of the saved state — without it, restore
    # fails with a "structures do not match" error.
    restored = mngr.restore(
        resolved_step,
        args=ocp.args.Composite(
            state=ocp.args.PyTreeRestore(target, partial_restore=True),
        ),
    )
    return cast("M", eqx.combine(restored["state"][which], static))


def flush(mngr: ocp.CheckpointManager) -> None:
    """Block until all queued saves on ``mngr`` have finalised on disk.

    Idempotent when nothing is in flight. Always call before exiting
    the training loop so async writes aren't abandoned on shutdown.
    """
    mngr.wait_until_finished()


def write_config(run_dir: Path, config: BaseModel) -> None:
    """Dump a resolved pydantic config to ``run_dir/config.yaml``.

    The sidecar lives at the run-dir root (not under ``step_*/``),
    so it survives both the gcsfuse local view and the Orbax gs://
    checkpoint tree.
    """
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

    ``config.yaml`` lives at the run-dir root. The sidecar is
    authoritative for model shape since the on-disk weights were
    produced under it. If the user-supplied ``fallback`` disagrees we
    warn under ``log_event`` and keep going; a genuinely mismatched
    shape would fail at restore time one call later. Missing sidecar
    falls back silently so hand-constructed test runs keep working.
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
    """Replace ``ShapeDtypeStruct`` leaves with zero-filled real arrays.

    Optax's ``optimizer.init`` walks the model with ``eqx.filter(m,
    eqx.is_inexact_array)`` — a filter that returns False for
    ``ShapeDtypeStruct``, so an abstract skeleton produces an empty
    opt_state. Materialising to host-side zeros keeps the abstract
    path's "no graph trace, no init draws" promise while letting
    ``opt_state_builder`` see a real tree.
    """
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
    """Construct a skeleton state tree mirroring what ``save_checkpoint`` wrote.

    Materialises the (possibly abstract) ``model_skeleton`` to real
    host-side zeros so ``opt_state_builder`` can run through
    :func:`optax.GradientTransformation.init` — which uses
    ``eqx.is_inexact_array`` internally and would otherwise see Nones
    at every array position under an abstract skeleton.
    """
    real_skeleton = _materialize_skeleton(model_skeleton)
    return {
        "model": real_skeleton,
        "ema": real_skeleton,
        "opt": opt_state_builder(real_skeleton),
        "key": jax.random.key(0),
    }


def _assert_step_counters_match(opt_state: object, meta_step: int) -> None:
    """Log a warning if optax step counter and ``meta.step`` disagree.

    The optimizer's internal ``count`` is incremented on every
    ``optimizer.update`` call; it should equal ``meta.step`` at save
    time. Drift signals a bug in step tracking (e.g. a save happening
    on a step where the optimizer was skipped via
    :func:`nanodiffusion.optimizer.apply_or_skip`). We warn rather
    than fail so a single non-finite-gradient skip doesn't abort the
    resume — the operator can investigate the log.
    """

    def _is_count_holder(x: object) -> bool:
        # ``tuple.count`` is a method; we want a non-callable ``count``
        # attribute (the jnp scalar in optax's ScaleByAdamState etc.).
        count = getattr(x, "count", None)
        return count is not None and not callable(count)

    counts: list[int] = []
    for leaf in jax.tree.leaves(opt_state, is_leaf=_is_count_holder):
        if not _is_count_holder(leaf):
            continue
        try:
            counts.append(int(leaf.count))
        except (TypeError, ValueError):
            continue
    if not counts:
        return
    if not all(c == counts[0] for c in counts):
        logger.warning(
            "checkpoint_step_counter_disagreement",
            counts=counts,
            meta_step=meta_step,
        )
        return
    actual = counts[0]
    if actual != meta_step:
        logger.warning(
            "checkpoint_step_counter_mismatch",
            opt_state_count=actual,
            meta_step=meta_step,
        )
