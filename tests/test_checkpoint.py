from __future__ import annotations

import json
from typing import TYPE_CHECKING, assert_type

import equinox as eqx
import jax
import numpy as np
import optax
import pytest

from nanodiffusion.checkpoint import (
    CheckpointMeta,
    load_checkpoint,
    load_model,
    save_checkpoint,
)
from nanodiffusion.constants import (
    EMA_FILENAME,
    LATEST_LINK_NAME,
    META_FILENAME,
    MODEL_FILENAME,
    OPT_STATE_FILENAME,
)
from nanodiffusion.data.cursors import PretrainCursor
from nanodiffusion.model import Transformer
from tests._helpers import inexact_leaves

if TYPE_CHECKING:
    from pathlib import Path

    from nanodiffusion.config import ModelConfig


def test_roundtrip_preserves_model_weights(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)
    ema_model = model
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    cursor = PretrainCursor(epoch=2, shard_idx=5, row_group_idx=7)
    save_checkpoint(
        tmp_path / "ckpt",
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        step=42,
        cursor=cursor,
    )

    # Build an independent skeleton so we really test deserialisation
    # rather than picking up the already-in-memory arrays.
    skeleton_key = jax.random.PRNGKey(123)
    model_skeleton = Transformer(small_config, key=skeleton_key)

    loaded_model, loaded_ema, _loaded_opt_state, meta = load_checkpoint(
        tmp_path / "ckpt",
        model_skeleton=model_skeleton,
        opt_state_builder=lambda m: optimizer.init(eqx.filter(m, eqx.is_inexact_array)),
    )

    # Generic narrowing: basedpyright must infer ``M = Transformer`` from
    # the skeleton, so the returned model/ema are typed ``Transformer``
    # and no downcast is needed at the call site.
    assert_type(loaded_model, Transformer)
    assert_type(loaded_ema, Transformer)
    assert type(loaded_model) is Transformer
    assert type(loaded_ema) is Transformer

    assert meta == CheckpointMeta(step=42, cursor=cursor)

    for a, b in zip(inexact_leaves(model), inexact_leaves(loaded_model), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(inexact_leaves(ema_model), inexact_leaves(loaded_ema), strict=True):
        np.testing.assert_array_equal(a, b)


def test_roundtrip_with_null_cursor(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """A fresh run with no prior cursor must round-trip as ``None``."""
    model = Transformer(small_config, key=key)
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    save_checkpoint(
        tmp_path / "ckpt",
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=0,
        cursor=None,
    )

    _loaded_model, _loaded_ema, _loaded_opt_state, meta = load_checkpoint(
        tmp_path / "ckpt",
        model_skeleton=model,
        opt_state_builder=lambda m: optimizer.init(eqx.filter(m, eqx.is_inexact_array)),
    )
    assert meta.step == 0
    assert meta.cursor is None


def test_save_writes_expected_files(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    model = Transformer(small_config, key=key)
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ckpt = tmp_path / "ckpt"
    save_checkpoint(
        ckpt,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=1,
        cursor=None,
    )
    files = {p.name for p in ckpt.iterdir()}
    assert {MODEL_FILENAME, EMA_FILENAME, OPT_STATE_FILENAME, META_FILENAME} <= files
    meta = json.loads((ckpt / META_FILENAME).read_text())
    assert meta["step"] == 1
    assert meta["cursor"] is None


def test_load_model_narrows_and_picks_snapshot(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """``load_model`` returns the skeleton's concrete type and honors ``which``."""
    key, model_key, ema_key = jax.random.split(key, 3)
    model = Transformer(small_config, key=model_key)
    ema_model = Transformer(small_config, key=ema_key)
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    save_checkpoint(
        tmp_path / "ckpt",
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        step=7,
        cursor=None,
    )

    model_skeleton = Transformer(small_config, key=jax.random.PRNGKey(999))

    loaded_ema = load_model(
        tmp_path / "ckpt", model_skeleton=model_skeleton, which="ema"
    )
    loaded_raw = load_model(
        tmp_path / "ckpt", model_skeleton=model_skeleton, which="model"
    )

    assert_type(loaded_ema, Transformer)
    assert_type(loaded_raw, Transformer)

    for a, b in zip(inexact_leaves(ema_model), inexact_leaves(loaded_ema), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(inexact_leaves(model), inexact_leaves(loaded_raw), strict=True):
        np.testing.assert_array_equal(a, b)


def _make_opt(
    model: Transformer,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    optimizer = optax.adamw(1e-3)
    return optimizer, optimizer.init(eqx.filter(model, eqx.is_inexact_array))


def test_save_atomic_rename_leaves_no_tmp_sidecar(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """Happy path: after a save, only the target exists (no ``.tmp`` dir)."""
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    save_checkpoint(
        tmp_path / "step_1",
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=1,
        cursor=None,
    )
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"step_1"}, f"unexpected entries: {names}"


def test_save_refuses_to_overwrite_existing_target(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """Double-save to the same path raises rather than clobbering state."""
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    save_checkpoint(
        tmp_path / "step_1",
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=1,
        cursor=None,
    )
    with pytest.raises(FileExistsError, match="step_1"):
        save_checkpoint(
            tmp_path / "step_1",
            model=model,
            ema_model=model,
            opt_state=opt_state,
            step=1,
            cursor=None,
        )


def test_save_cleans_up_stale_tmp_sibling(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """A crash-leftover ``step_1.tmp`` from a prior run must not block the next save."""
    stale = tmp_path / "step_1.tmp"
    stale.mkdir()
    (stale / "garbage").write_text("leftover")

    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    save_checkpoint(
        tmp_path / "step_1",
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=1,
        cursor=None,
    )
    assert (tmp_path / "step_1" / META_FILENAME).exists()
    assert not stale.exists()


def test_update_latest_symlink_points_at_newest_checkpoint(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """Two consecutive ``update_latest`` saves leave ``latest`` on the newer one."""
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    save_checkpoint(
        tmp_path / "step_1",
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=1,
        cursor=None,
        update_latest=True,
    )
    save_checkpoint(
        tmp_path / "step_2",
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=2,
        cursor=None,
        update_latest=True,
    )

    latest = tmp_path / LATEST_LINK_NAME
    assert latest.is_symlink()
    # Relative target — stays valid if the run dir is moved.
    assert str(latest.readlink()) == "step_2"
    # Loading through the symlink resolves to the newest meta.
    meta = CheckpointMeta.model_validate_json((latest / META_FILENAME).read_text())
    assert meta.step == 2


def test_update_latest_default_off(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """Without ``update_latest`` the symlink is never created."""
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    save_checkpoint(
        tmp_path / "step_1",
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=1,
        cursor=None,
    )
    assert not (tmp_path / LATEST_LINK_NAME).exists()
    assert not (tmp_path / LATEST_LINK_NAME).is_symlink()


def test_load_checkpoint_follows_latest_symlink(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """``load_checkpoint(run_dir/'latest')`` behaves identically to the direct path."""
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)
    _, opt_state = _make_opt(model)

    save_checkpoint(
        tmp_path / "step_7",
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=7,
        cursor=None,
        update_latest=True,
    )

    skeleton = Transformer(small_config, key=jax.random.PRNGKey(321))
    optimizer, _ = _make_opt(skeleton)
    _m, _e, _o, meta = load_checkpoint(
        tmp_path / LATEST_LINK_NAME,
        model_skeleton=skeleton,
        opt_state_builder=lambda m: optimizer.init(eqx.filter(m, eqx.is_inexact_array)),
    )
    assert meta.step == 7


def test_checkpoint_meta_rejects_negative_step() -> None:
    """Pydantic validation catches nonsense step values on load."""
    with pytest.raises(ValueError, match="step"):
        CheckpointMeta(step=-1, cursor=None)


def test_checkpoint_meta_json_round_trip() -> None:
    """``model_dump_json`` output parses back into an equal model."""
    cursor = PretrainCursor(epoch=1, shard_idx=2, row_group_idx=3)
    meta = CheckpointMeta(step=10, cursor=cursor)
    blob = meta.model_dump_json()
    assert CheckpointMeta.model_validate_json(blob) == meta
    # Confirm the on-wire shape hasn't drifted (downstream tooling greps it).
    parsed = json.loads(blob)
    assert parsed == {
        "step": 10,
        "cursor": {
            "kind": "pretrain",
            "epoch": 1,
            "shard_idx": 2,
            "row_group_idx": 3,
        },
    }
