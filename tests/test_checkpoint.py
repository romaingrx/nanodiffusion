from __future__ import annotations

import json
from typing import TYPE_CHECKING, assert_type

import equinox as eqx
import jax
import numpy as np
import optax

from nanodiffusion.checkpoint import (
    CheckpointMeta,
    load_checkpoint,
    load_model,
    save_checkpoint,
)
from nanodiffusion.model import Transformer

if TYPE_CHECKING:
    from pathlib import Path

    from nanodiffusion.config import ModelConfig
    from nanodiffusion.data.source import SourcePosition


def _leaves(m: Transformer) -> list[jax.Array]:
    return jax.tree.leaves(eqx.filter(m, eqx.is_inexact_array))


def test_roundtrip_preserves_model_weights(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)
    ema_model = model
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    cursor: SourcePosition = {"epoch": 2, "shard_idx": 5, "row_group_idx": 7}
    save_checkpoint(
        tmp_path / "ckpt",
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        step=42,
        cursor=cursor,
    )

    # Build independent skeletons so we really test deserialisation rather
    # than picking up the already-in-memory arrays.
    skeleton_key = jax.random.PRNGKey(123)
    model_skeleton = Transformer(small_config, key=skeleton_key)
    opt_state_skeleton = optimizer.init(
        eqx.filter(model_skeleton, eqx.is_inexact_array)
    )

    loaded_model, loaded_ema, _loaded_opt_state, meta = load_checkpoint(
        tmp_path / "ckpt",
        model_skeleton=model_skeleton,
        opt_state_skeleton=opt_state_skeleton,
    )

    # Generic narrowing: basedpyright must infer ``M = Transformer`` from
    # the skeleton, so the returned model/ema are typed ``Transformer``
    # and no downcast is needed at the call site.
    assert_type(loaded_model, Transformer)
    assert_type(loaded_ema, Transformer)
    assert type(loaded_model) is Transformer
    assert type(loaded_ema) is Transformer

    assert meta == CheckpointMeta(step=42, cursor=cursor)

    for a, b in zip(_leaves(model), _leaves(loaded_model), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(_leaves(ema_model), _leaves(loaded_ema), strict=True):
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
        opt_state_skeleton=opt_state,
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
    assert {"model.eqx", "ema.eqx", "opt_state.eqx", "meta.json"} <= files
    meta = json.loads((ckpt / "meta.json").read_text())
    assert meta["step"] == 1
    assert meta["cursor"] is None


def test_load_model_narrows_and_picks_snapshot(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """``load_model`` returns the skeleton's concrete type and honors ``which``."""
    key, model_key, ema_key = jax.random.split(key, 3)
    model = Transformer(small_config, key=model_key)
    # Construct a distinguishable EMA so we can tell which file was read.
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

    for a, b in zip(_leaves(ema_model), _leaves(loaded_ema), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(_leaves(model), _leaves(loaded_raw), strict=True):
        np.testing.assert_array_equal(a, b)
