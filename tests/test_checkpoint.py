from __future__ import annotations

import inspect
import json
import threading
import time
from typing import TYPE_CHECKING, assert_type

import equinox as eqx
import jax
import numpy as np
import optax
import pytest
import structlog.testing

from nanodiffusion import checkpoint as ckpt_mod
from nanodiffusion.checkpoint import (
    CheckpointMeta,
    flush,
    load_checkpoint,
    load_meta,
    load_model,
    make_manager,
    save_checkpoint,
)
from nanodiffusion.data.cursors import PretrainCursor
from nanodiffusion.model import Transformer
from tests._helpers import inexact_leaves

if TYPE_CHECKING:
    from pathlib import Path

    from nanodiffusion.config import ModelConfig


# Orbax writes ``_CHECKPOINT_METADATA`` atomically as the final finalisation
# step on both POSIX and object stores; its presence is the durable marker.
_FINALISED_MARKER = "_CHECKPOINT_METADATA"


def _make_opt(
    model: Transformer,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    optimizer = optax.adamw(1e-3)
    return optimizer, optimizer.init(eqx.filter(model, eqx.is_inexact_array))


def _opt_builder(
    optimizer: optax.GradientTransformation,
) -> object:
    def build(m: Transformer) -> optax.OptState:
        return optimizer.init(eqx.filter(m, eqx.is_inexact_array))

    return build


def test_roundtrip_preserves_model_weights(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)
    ema_model = model
    optimizer, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path)
    cursor = PretrainCursor(
        epoch=2, shard_idx=5, row_group_idx=7, doc_idx=11, token_offset=13
    )
    save_checkpoint(
        mngr,
        42,
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=cursor,
    )
    flush(mngr)

    skeleton_key = jax.random.PRNGKey(123)
    model_skeleton = Transformer(small_config, key=skeleton_key)

    loaded_model, loaded_ema, _loaded_opt_state, _loaded_key, meta = load_checkpoint(
        mngr,
        model_skeleton=model_skeleton,
        opt_state_builder=_opt_builder(optimizer),
    )

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
    model = Transformer(small_config, key=key)
    optimizer, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path)
    save_checkpoint(
        mngr,
        0,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )
    flush(mngr)

    _m, _e, _o, _k, meta = load_checkpoint(
        mngr,
        model_skeleton=model,
        opt_state_builder=_opt_builder(optimizer),
    )
    assert meta.step == 0
    assert meta.cursor is None


def test_save_writes_orbax_layout(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path)
    save_checkpoint(
        mngr,
        1,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )
    flush(mngr)

    step_dir = tmp_path / "step_1"
    assert step_dir.is_dir()
    assert (step_dir / "state").is_dir()
    assert (step_dir / "meta").is_dir()
    assert (step_dir / _FINALISED_MARKER).exists()


def test_load_model_narrows_and_picks_snapshot(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    key, model_key, ema_key = jax.random.split(key, 3)
    model = Transformer(small_config, key=model_key)
    ema_model = Transformer(small_config, key=ema_key)
    _, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path)
    save_checkpoint(
        mngr,
        7,
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )
    flush(mngr)

    model_skeleton = Transformer(small_config, key=jax.random.PRNGKey(999))

    loaded_ema = load_model(tmp_path, model_skeleton=model_skeleton, which="ema")
    loaded_raw = load_model(tmp_path, model_skeleton=model_skeleton, which="model")

    assert_type(loaded_ema, Transformer)
    assert_type(loaded_raw, Transformer)

    for a, b in zip(inexact_leaves(ema_model), inexact_leaves(loaded_ema), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(inexact_leaves(model), inexact_leaves(loaded_raw), strict=True):
        np.testing.assert_array_equal(a, b)


def test_load_checkpoint_uses_latest_step(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    model = Transformer(small_config, key=key)
    optimizer, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path)
    for step in (1, 3, 2):
        save_checkpoint(
            mngr,
            step,
            model=model,
            ema_model=model,
            opt_state=opt_state,
            key=jax.random.key(0),
            cursor=None,
        )
    flush(mngr)

    _m, _e, _o, _k, meta = load_checkpoint(
        tmp_path,
        model_skeleton=model,
        opt_state_builder=_opt_builder(optimizer),
    )
    assert meta.step == 3


def test_max_to_keep_garbage_collects_old_steps(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path, max_to_keep=2)
    for step in (1, 2, 3, 4, 5):
        save_checkpoint(
            mngr,
            step,
            model=model,
            ema_model=model,
            opt_state=opt_state,
            key=jax.random.key(0),
            cursor=None,
        )
    flush(mngr)

    remaining = sorted(
        int(p.name.removeprefix("step_"))
        for p in tmp_path.iterdir()
        if p.is_dir() and p.name.startswith("step_")
    )
    assert remaining == [4, 5]


def test_ema_weights_stay_distinct_from_model_after_restore(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """Catches accidental aliasing through the dict->partition->combine path."""
    key, m_key, e_key = jax.random.split(key, 3)
    model = Transformer(small_config, key=m_key)
    ema_model = Transformer(small_config, key=e_key)
    optimizer, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path)
    save_checkpoint(
        mngr,
        1,
        model=model,
        ema_model=ema_model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )
    flush(mngr)

    skeleton = Transformer(small_config, key=jax.random.PRNGKey(0))
    loaded_model, loaded_ema, _o, _k, _meta = load_checkpoint(
        mngr,
        model_skeleton=skeleton,
        opt_state_builder=_opt_builder(optimizer),
    )

    model_leaves = inexact_leaves(loaded_model)
    ema_leaves = inexact_leaves(loaded_ema)
    assert len(model_leaves) == len(ema_leaves)
    differ = any(
        not np.array_equal(a, b) for a, b in zip(model_leaves, ema_leaves, strict=True)
    )
    assert differ, "EMA and model should restore with different weights"


def test_roundtrip_preserves_rng_key(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    model = Transformer(small_config, key=key)
    optimizer, opt_state = _make_opt(model)

    saved_key = jax.random.fold_in(jax.random.key(0), 12345)
    mngr = make_manager(tmp_path)
    save_checkpoint(
        mngr,
        42,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=saved_key,
        cursor=None,
    )
    flush(mngr)

    _m, _e, _o, loaded_key, _meta = load_checkpoint(
        mngr,
        model_skeleton=model,
        opt_state_builder=_opt_builder(optimizer),
    )
    np.testing.assert_array_equal(
        jax.random.uniform(loaded_key, (8,)),
        jax.random.uniform(saved_key, (8,)),
    )


def test_save_has_no_rank_zero_gate(small_config: ModelConfig, key: jax.Array) -> None:
    """A Python-level rank-0 gate would deadlock real multi-host runs, since
    Orbax does its own internal coordination. We can't reproduce multi-host
    on a single-process test, so assert the source has no such gate."""
    src = inspect.getsource(ckpt_mod.save_checkpoint)
    assert "process_index" not in src, (
        "save_checkpoint must not gate on jax.process_index — "
        f"found a reference in the implementation:\n{src}"
    )
    _ = small_config, key


def test_load_meta_reads_just_the_metadata(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)
    cursor = PretrainCursor(
        epoch=1, shard_idx=2, row_group_idx=3, doc_idx=4, token_offset=5
    )

    mngr = make_manager(tmp_path)
    save_checkpoint(
        mngr,
        9,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=cursor,
    )
    flush(mngr)

    meta = load_meta(tmp_path)
    assert meta == CheckpointMeta(step=9, cursor=cursor)


def test_checkpoint_meta_rejects_negative_step() -> None:
    with pytest.raises(ValueError, match="step"):
        CheckpointMeta(step=-1, cursor=None)


def test_checkpoint_meta_rejects_legacy_pretrain_cursor() -> None:
    """Legacy row-group-only cursors are ambiguous under exact-resume semantics."""
    legacy = {
        "step": 10,
        "cursor": {
            "kind": "pretrain",
            "epoch": 1,
            "shard_idx": 2,
            "row_group_idx": 3,
        },
    }
    with pytest.raises(ValueError, match="doc_idx"):
        CheckpointMeta.model_validate(legacy)


def test_back_to_back_saves_serialise_via_queue(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    """A second save blocks on the first via Orbax's queue-depth-1 backpressure."""
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path)
    save_checkpoint(
        mngr,
        1,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )
    save_checkpoint(
        mngr,
        2,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )
    assert (tmp_path / "step_1" / _FINALISED_MARKER).exists()
    flush(mngr)
    assert (tmp_path / "step_2" / _FINALISED_MARKER).exists()


def test_flush_blocks_until_durable(
    tmp_path: Path, small_config: ModelConfig, key: jax.Array
) -> None:
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path)
    save_checkpoint(
        mngr,
        1,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )

    flushed = threading.Event()

    def _flush_then_signal() -> None:
        flush(mngr)
        flushed.set()

    threading.Thread(target=_flush_then_signal, daemon=True).start()
    assert flushed.wait(timeout=10.0), "flush never returned"
    assert (tmp_path / "step_1" / _FINALISED_MARKER).exists()


def test_load_checkpoint_errors_when_no_step_exists(tmp_path: Path) -> None:
    mngr = make_manager(tmp_path)
    with pytest.raises(FileNotFoundError, match="no finalised checkpoint"):
        load_checkpoint(
            mngr,
            model_skeleton=None,
            opt_state_builder=lambda _m: None,
        )


def test_step_counter_reconciliation_warns_on_mismatch(
    tmp_path: Path,
    small_config: ModelConfig,
    key: jax.Array,
) -> None:
    model = Transformer(small_config, key=key)
    optimizer, opt_state = _make_opt(model)
    mngr = make_manager(tmp_path)
    save_checkpoint(
        mngr,
        42,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )
    flush(mngr)

    with structlog.testing.capture_logs() as cap_logs:
        _ = load_checkpoint(
            mngr,
            model_skeleton=model,
            opt_state_builder=_opt_builder(optimizer),
        )

    events = [r.get("event") for r in cap_logs]
    assert "checkpoint_step_counter_mismatch" in events, (
        f"expected mismatch warning, got events: {events}"
    )


def test_save_returns_before_flush_completes(
    tmp_path: Path,
    small_config: ModelConfig,
    key: jax.Array,
) -> None:
    """Submit (jax.device_get + mngr.save) is host-side; flush blocks on OCDBT
    upload. Even on local fs where both are fast, submit can't be slower."""
    model = Transformer(small_config, key=key)
    _, opt_state = _make_opt(model)

    mngr = make_manager(tmp_path)

    t0 = time.perf_counter()
    save_checkpoint(
        mngr,
        1,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )
    t_submit = time.perf_counter() - t0

    flush(mngr)
    t_total = time.perf_counter() - t0

    assert t_submit <= t_total


def test_checkpoint_meta_json_round_trip() -> None:
    cursor = PretrainCursor(
        epoch=1, shard_idx=2, row_group_idx=3, doc_idx=0, token_offset=0
    )
    meta = CheckpointMeta(step=10, cursor=cursor)
    blob = meta.model_dump_json()
    assert CheckpointMeta.model_validate_json(blob) == meta
    parsed = json.loads(blob)
    assert parsed == {
        "step": 10,
        "cursor": {
            "kind": "pretrain",
            "epoch": 1,
            "shard_idx": 2,
            "row_group_idx": 3,
            "doc_idx": 0,
            "token_offset": 0,
        },
    }
