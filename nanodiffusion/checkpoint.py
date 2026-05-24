"""On-disk checkpoints for a training run.

A checkpoint directory holds four equinox-serialised binary files
(``model.eqx`` / ``ema.eqx`` / ``opt_state.eqx`` / ``rng.eqx``) plus a
``meta.json`` sidecar for the step / data-loader cursor and a
``config.yaml`` written by :func:`write_config`. The RNG key lives in
its own binary artifact so resume continues the same stochastic chain
(masking / timestep sampling) rather than rewinding to the seed.

Binary save/load is generic over the concrete diffusion-model class:
pass a ``Transformer`` skeleton in, get a ``Transformer`` back, no
casts at the call site. Only the sidecar helpers touch ``Config``, so
a pydantic schema change can only drift through that path.
"""

import shutil
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, cast

import equinox as eqx
import jax
import optax
import structlog
import yaml
from jax.experimental import multihost_utils
from pydantic import BaseModel, ConfigDict, Field

from nanodiffusion.config import Config, ModelConfig
from nanodiffusion.constants import (
    CONFIG_SIDECAR_FILENAME,
    EMA_FILENAME,
    LATEST_LINK_NAME,
    META_FILENAME,
    MODEL_FILENAME,
    OPT_STATE_FILENAME,
    RNG_FILENAME,
)
from nanodiffusion.data.cursors import LoaderCursor
from nanodiffusion.model import DiffusionModel
from nanodiffusion.types import PRNGKeyArray

logger = structlog.get_logger(__name__)

type ModelSnapshot = Literal["model", "ema"]


class _AsyncSaveState:
    """Mutable singleton for the async-save executor and in-flight future.

    Class attributes (not module-level ``global`` rebinding) so writers
    just assign and the lint stays clean. One worker, queue depth one:
    a slow gcsfuse upload backpressures the next call to
    :func:`save_checkpoint_async`, never spawns a second snapshot.
    """

    executor: ThreadPoolExecutor | None = None
    inflight: Future[None] | None = None


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


def save_checkpoint[M: DiffusionModel](
    path: Path,
    *,
    model: M,
    ema_model: M,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
    step: int,
    cursor: LoaderCursor | None,
    update_latest: bool = False,
) -> None:
    """Write ``(model, ema, opt_state, rng, meta)`` atomically to ``path``.

    Synchronous: the training thread blocks until all four equinox
    leaves are serialised and the ``.tmp`` dir renames into place. Use
    :func:`save_checkpoint_async` from the training loop when the
    serialise + upload cost matters (gcsfuse cross-region uploads can
    run into minutes); the sync path is kept for tests and inference
    dumps where a deterministic write is the simpler contract.

    Multi-host safe: only the rank-0 process writes; all hosts then
    barrier on :func:`multihost_utils.sync_global_devices` so no reader
    races ahead of the on-disk artifact. Safe to call unconditionally
    from every host — the gate lives inside rather than at call sites.

    Leaves land in a sibling ``<path>.tmp`` dir first, then a single
    :func:`os.replace` renames into place (atomic on POSIX; on
    gcsfuse-backed paths the rename maps to a GCS copy+delete which is
    not strictly atomic, so partial ``.tmp`` dirs are possible after a
    hard preemption and get cleaned up on the next save). Raises
    :class:`FileExistsError` if ``path`` already exists — we refuse to
    silently overwrite a prior checkpoint.

    When ``update_latest=True``, point ``path.parent/latest`` at
    ``path.name`` via a temp-link swap; platforms that reject symlinks
    (unprivileged Windows) log and skip rather than failing the save.
    """
    if jax.process_index() == 0:
        _write_snapshot(
            path,
            snapshot=(model, ema_model, opt_state, key),
            step=step,
            cursor=cursor,
            config_yaml=None,
            update_latest=update_latest,
        )
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices(f"save_checkpoint:{path.name}")


def save_checkpoint_async[M: DiffusionModel](
    path: Path,
    *,
    model: M,
    ema_model: M,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
    step: int,
    cursor: LoaderCursor | None,
    config: BaseModel | None = None,
    update_latest: bool = False,
) -> None:
    """Snapshot ``(model, ema, opt_state, rng)`` and write it on a worker thread.

    The :func:`jax.device_get` snapshot decouples the artifact from
    device-resident arrays before returning, so the caller can mutate
    params on the next training step while the host-side write is still
    in flight. The executor is a single worker — bounded queue depth of
    one — so a slow upload backpressures the next save rather than
    spawning unbounded snapshots.

    Pass ``config`` to bundle ``config.yaml`` into the same atomic
    rename; the YAML is rendered on the calling thread (cheap) so the
    worker thread doesn't touch the live config object. Multi-host
    semantics match :func:`save_checkpoint`: rank-0 writes, all hosts
    barrier — here on a ``snap:<name>`` token so the barrier reflects
    the moment the snapshot is host-resident, not when the write lands.
    """
    if _AsyncSaveState.inflight is not None:
        _AsyncSaveState.inflight.result()
        _AsyncSaveState.inflight = None

    is_rank_zero = jax.process_index() == 0
    snapshot: tuple[M, M, optax.OptState, PRNGKeyArray] | None = None
    config_yaml: str | None = None
    if is_rank_zero:
        snapshot = jax.device_get((model, ema_model, opt_state, key))
        if config is not None:
            config_yaml = yaml.dump(config.model_dump(mode="json"))

    if jax.process_count() > 1:
        multihost_utils.sync_global_devices(f"snap:{path.name}")

    if not is_rank_zero or snapshot is None:
        return

    if _AsyncSaveState.executor is None:
        _AsyncSaveState.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt-save"
        )
    _AsyncSaveState.inflight = _AsyncSaveState.executor.submit(
        _write_snapshot,
        path,
        snapshot=snapshot,
        step=step,
        cursor=cursor,
        config_yaml=config_yaml,
        update_latest=update_latest,
    )


def flush_pending_save() -> None:
    """Block on any in-flight :func:`save_checkpoint_async` write.

    Re-raises whatever exception the worker raised so caller sees a
    failed save rather than silently losing the last checkpoint. Idle
    when no save is in flight, including on non-rank-0 hosts where
    submission is skipped.
    """
    if _AsyncSaveState.inflight is not None:
        try:
            _AsyncSaveState.inflight.result()
        finally:
            _AsyncSaveState.inflight = None


def _write_snapshot[M: DiffusionModel](
    path: Path,
    *,
    snapshot: tuple[M, M, optax.OptState, PRNGKeyArray],
    step: int,
    cursor: LoaderCursor | None,
    config_yaml: str | None,
    update_latest: bool,
) -> None:
    model, ema_model, opt_state, key = snapshot
    if path.exists():
        msg = f"refusing to overwrite existing checkpoint at {path}"
        raise FileExistsError(msg)

    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    eqx.tree_serialise_leaves(tmp_path / MODEL_FILENAME, model)
    eqx.tree_serialise_leaves(tmp_path / EMA_FILENAME, ema_model)
    eqx.tree_serialise_leaves(tmp_path / OPT_STATE_FILENAME, opt_state)
    eqx.tree_serialise_leaves(tmp_path / RNG_FILENAME, key)
    meta = CheckpointMeta(step=step, cursor=cursor)
    (tmp_path / META_FILENAME).write_text(meta.model_dump_json(indent=2))
    if config_yaml is not None:
        (tmp_path / CONFIG_SIDECAR_FILENAME).write_text(config_yaml)

    tmp_path.replace(path)

    if update_latest:
        _update_latest_symlink(path.parent, path.name)


def _update_latest_symlink(run_dir: Path, target_name: str) -> None:
    """Atomically point ``run_dir/latest`` at ``target_name``.

    Uses a relative target so the run directory is portable, and
    renames a temp link over the existing one so readers never
    observe a missing ``latest``. ``OSError`` is logged and swallowed
    for filesystems that reject symlinks.
    """
    tmp_link = run_dir / (LATEST_LINK_NAME + ".tmp")
    final_link = run_dir / LATEST_LINK_NAME
    try:
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        tmp_link.symlink_to(target_name)
        tmp_link.replace(final_link)
    except OSError as exc:
        logger.warning(
            "latest_symlink_skipped",
            run_dir=str(run_dir),
            target=target_name,
            error=str(exc),
        )


def load_checkpoint[M: DiffusionModel](
    path: Path,
    *,
    model_skeleton: M,
    opt_state_builder: Callable[[M], optax.OptState],
) -> tuple[M, M, optax.OptState, PRNGKeyArray, CheckpointMeta]:
    """Inverse of :func:`save_checkpoint`.

    ``model_skeleton`` may be either a real :class:`DiffusionModel` or
    an abstract shape tree from :func:`eqx.filter_eval_shape`; only
    leaf shape + dtype are read. ``opt_state_builder`` is called with
    the freshly loaded model to shape the opt-state tree — typically
    ``lambda m: optimizer.init(eqx.filter(m, eqx.is_inexact_array))``.
    Threading a builder instead of a pre-built skeleton lets callers
    skip the wasteful real-weights init on resume and keeps the
    two-phase load hidden from every driver.

    ``path`` may be a ``latest`` symlink; the filesystem resolves it
    transparently. The return type is bound to the skeleton's concrete
    type so callers keep model-specific attributes without downcasts.
    """
    model = cast(
        "M", eqx.tree_deserialise_leaves(path / MODEL_FILENAME, model_skeleton)
    )
    ema_model = cast(
        "M", eqx.tree_deserialise_leaves(path / EMA_FILENAME, model_skeleton)
    )
    opt_state_skeleton = opt_state_builder(model)
    opt_state = cast(
        "optax.OptState",
        eqx.tree_deserialise_leaves(path / OPT_STATE_FILENAME, opt_state_skeleton),
    )
    key_skeleton = jax.random.PRNGKey(0)
    key = cast(
        "PRNGKeyArray",
        eqx.tree_deserialise_leaves(path / RNG_FILENAME, key_skeleton),
    )
    meta = CheckpointMeta.model_validate_json((path / META_FILENAME).read_text())
    return model, ema_model, opt_state, key, meta


def load_model[M: DiffusionModel](
    path: Path,
    *,
    model_skeleton: M,
    which: ModelSnapshot = "ema",
) -> M:
    """Read just the model weights from a training checkpoint.

    For inference and sampling paths that don't need optimizer state
    or metadata. ``which="ema"`` (more stable, sampling default) or
    ``"model"`` (raw trained weights).
    """
    filename = EMA_FILENAME if which == "ema" else MODEL_FILENAME
    return cast("M", eqx.tree_deserialise_leaves(path / filename, model_skeleton))


def write_config(run_dir: Path, config: BaseModel) -> None:
    """Dump a resolved pydantic config to ``run_dir/config.yaml``.

    Uses ``model_dump(mode="json")`` so ``Path`` and other non-yaml
    types round-trip cleanly through :func:`yaml.safe_load`.
    """
    (run_dir / CONFIG_SIDECAR_FILENAME).write_text(
        yaml.dump(config.model_dump(mode="json"))
    )


def resolve_model_config_from_checkpoint(
    checkpoint: Path,
    *,
    fallback: ModelConfig,
    log_event: str,
) -> ModelConfig:
    """Pick up a checkpoint's ``config.yaml`` model section when present.

    The sidecar is authoritative for model shape since the on-disk
    weights were produced under it. If the user-supplied ``fallback``
    disagrees we warn under ``log_event`` and keep going; a genuinely
    mismatched shape would fail at deserialisation one line later.
    Missing sidecar falls back silently so hand-constructed test
    checkpoints keep working.
    """
    sidecar = checkpoint / CONFIG_SIDECAR_FILENAME
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
