"""On-disk checkpoints for a pretraining run.

A checkpoint is a directory holding four files:

* ``model.eqx`` / ``ema.eqx`` — equinox-serialised diffusion models
* ``opt_state.eqx`` — equinox-serialised optax state (matches the shape
  produced by :func:`nanodiffusion.train.make_optimizer`)
* ``meta.json`` — step counter and data-loader cursor for clean resume

The public API is generic over the concrete diffusion-model class so the
skeleton type flows straight through to the return of
:func:`load_checkpoint`: pass a ``Transformer`` skeleton in, get a
``Transformer`` back — no casts at the call site.

The module is deliberately config-agnostic: the training loop owns
``config.yaml`` so saving and loading state never re-encodes user config
and stays out of the pydantic-versioning rabbit hole.
"""

import shutil
from pathlib import Path
from typing import Literal, cast

import equinox as eqx
import optax
import structlog
from pydantic import BaseModel, ConfigDict, Field

from nanodiffusion.data.source import SourcePosition
from nanodiffusion.model import DiffusionModel

logger = structlog.get_logger(__name__)

type ModelSnapshot = Literal["model", "ema"]

LATEST_LINK_NAME = "latest"


class CheckpointMeta(BaseModel):
    """Step counter + data-loader cursor persisted alongside weights.

    Pydantic handles json round-trip so neither the ``step`` coercion nor
    the ``cursor`` TypedDict shape check live in save/load. ``frozen=True``
    mirrors the previous dataclass contract: callers treat meta as a
    value object.
    """

    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0)
    cursor: SourcePosition | None = None


def save_checkpoint[M: DiffusionModel](
    path: Path,
    *,
    model: M,
    ema_model: M,
    opt_state: optax.OptState,
    step: int,
    cursor: SourcePosition | None,
    update_latest: bool = False,
) -> None:
    """Write ``(model, ema, opt_state, meta)`` atomically to ``path``.

    Files are first written under a sibling ``<path>.tmp`` directory;
    once every leaf has landed the directory is renamed into place via
    :func:`os.replace`, which is atomic on POSIX. A crash mid-write
    leaves only the ``.tmp`` sibling, never a half-populated ``path``.

    Raises :class:`FileExistsError` if ``path`` already exists — we
    refuse to silently overwrite a prior checkpoint. Lingering ``.tmp``
    siblings from a previous crash are cleaned up before the fresh write.

    When ``update_latest=True``, atomically point
    ``path.parent/latest`` at ``path.name`` after the rename. The
    symlink is updated via ``os.symlink`` + :func:`os.replace` on a
    temp link, so readers never see a dangling target. On platforms
    that reject symlinks (e.g. unprivileged Windows) the update is
    logged and skipped rather than failing the save.

    ``model`` and ``ema_model`` are constrained to the same concrete
    subtype of :class:`DiffusionModel`, so a typo like swapping the EMA
    for an unrelated module is a type error, not a silent disk write.
    """
    if path.exists():
        msg = f"refusing to overwrite existing checkpoint at {path}"
        raise FileExistsError(msg)

    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    eqx.tree_serialise_leaves(tmp_path / "model.eqx", model)
    eqx.tree_serialise_leaves(tmp_path / "ema.eqx", ema_model)
    eqx.tree_serialise_leaves(tmp_path / "opt_state.eqx", opt_state)
    meta = CheckpointMeta(step=step, cursor=cursor)
    (tmp_path / "meta.json").write_text(meta.model_dump_json(indent=2))

    tmp_path.replace(path)

    if update_latest:
        _update_latest_symlink(path.parent, path.name)


def _update_latest_symlink(run_dir: Path, target_name: str) -> None:
    """Atomically point ``run_dir/latest`` at ``target_name``.

    Uses a relative target so the run directory is portable. The new
    symlink is created at a temp name and renamed over the existing
    ``latest`` link — :meth:`Path.replace` is atomic on POSIX, so a
    reader never observes a missing ``latest`` after the first
    successful update. ``OSError`` is caught and logged so an
    unprivileged Windows host degrades gracefully instead of aborting
    training.
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
    opt_state_skeleton: optax.OptState,
) -> tuple[M, M, optax.OptState, CheckpointMeta]:
    """Inverse of :func:`save_checkpoint`.

    Skeletons must match the saved shapes; callers normally build them
    by reconstructing a fresh model and calling ``optimizer.init`` on it
    before deserialising. The return is bound to the skeleton's concrete
    type via the ``M`` type var, so callers can keep using model-specific
    attributes without downcasts.

    ``path`` may be a ``latest`` symlink; the filesystem resolves it
    transparently for both the equinox file reads and the meta parse.

    The ``cast`` calls mirror equinox's own ``_ad.py:185`` pattern: the
    underlying ``tree_deserialise_leaves`` is typed to return ``PyTree``,
    which we know structurally matches the skeleton we passed in.
    """
    model = cast("M", eqx.tree_deserialise_leaves(path / "model.eqx", model_skeleton))
    ema_model = cast("M", eqx.tree_deserialise_leaves(path / "ema.eqx", model_skeleton))
    opt_state = cast(
        "optax.OptState",
        eqx.tree_deserialise_leaves(path / "opt_state.eqx", opt_state_skeleton),
    )
    meta = CheckpointMeta.model_validate_json((path / "meta.json").read_text())
    return model, ema_model, opt_state, meta


def load_model[M: DiffusionModel](
    path: Path,
    *,
    model_skeleton: M,
    which: ModelSnapshot = "ema",
) -> M:
    """Read just the model weights from a training checkpoint.

    For inference and sampling pipelines that don't need optimizer
    state or metadata. ``which`` picks the snapshot file — ``"ema"``
    (the sampling default, more stable) or ``"model"`` (raw trained
    weights). The return is bound to the skeleton's concrete type via
    the ``M`` type var, so the caller keeps full type information
    without a downcast.
    """
    filename = "ema.eqx" if which == "ema" else "model.eqx"
    return cast("M", eqx.tree_deserialise_leaves(path / filename, model_skeleton))
