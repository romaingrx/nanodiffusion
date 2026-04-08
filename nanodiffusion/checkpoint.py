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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import equinox as eqx
import optax

from nanodiffusion.data.source import SourcePosition
from nanodiffusion.model import DiffusionModel

type ModelSnapshot = Literal["model", "ema"]


@dataclass(frozen=True, slots=True)
class CheckpointMeta:
    step: int
    cursor: SourcePosition | None


def save_checkpoint[M: DiffusionModel](
    path: Path,
    *,
    model: M,
    ema_model: M,
    opt_state: optax.OptState,
    step: int,
    cursor: SourcePosition | None,
) -> None:
    """Write ``(model, ema, opt_state, step, cursor)`` atomically-enough.

    The directory is created if missing. Individual file writes are not
    transactional; a crash mid-write can leave a partial checkpoint. A
    caller that needs safer semantics can write to a temp dir and rename.

    ``model`` and ``ema_model`` are constrained to the same concrete
    subtype of :class:`DiffusionModel`, so a typo like swapping the EMA
    for an unrelated module is a type error, not a silent disk write.
    """
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / "model.eqx", model)
    eqx.tree_serialise_leaves(path / "ema.eqx", ema_model)
    eqx.tree_serialise_leaves(path / "opt_state.eqx", opt_state)
    (path / "meta.json").write_text(
        json.dumps({"step": step, "cursor": cursor}, indent=2)
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
    meta_data = json.loads((path / "meta.json").read_text())
    cursor_raw = meta_data.get("cursor")
    cursor: SourcePosition | None = cursor_raw if cursor_raw is not None else None
    meta = CheckpointMeta(step=int(meta_data["step"]), cursor=cursor)
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
