from typing import assert_type

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pydantic import ValidationError

from nanodiffusion.config import ModelConfig, TrainConfig
from nanodiffusion.model import Transformer
from nanodiffusion.optimizer import ema_update, make_optimizer
from nanodiffusion.pretrain.train import TrainStepFn, make_train_step
from nanodiffusion.schedule import LogLinearSchedule
from tests._helpers import clone_state, inexact_leaves


def test_make_optimizer_warmup_peak_and_decay() -> None:
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=100, max_steps=1000)
    _opt, sched = make_optimizer(cfg)

    assert float(sched(0)) == pytest.approx(0.0, abs=1e-8)
    assert float(sched(100)) == pytest.approx(1e-3, rel=1e-3)
    assert float(sched(1000)) < 1e-5


def _count_inexact_leaves_that_changed(
    before: Transformer, after: Transformer, *, atol: float = 0.0
) -> int:
    return sum(
        1
        for a, b in zip(inexact_leaves(before), inexact_leaves(after), strict=True)
        if not np.allclose(a, b, atol=atol)
    )


def test_ema_update_at_decay_zero_copies_model(model: Transformer) -> None:
    """With decay=0, ema becomes an exact copy of ``model``."""
    perturbed = jax.tree.map(lambda x: x + 1.0 if eqx.is_inexact_array(x) else x, model)

    updated = ema_update(model, perturbed, decay=0.0)

    for x, y in zip(inexact_leaves(updated), inexact_leaves(perturbed), strict=True):
        np.testing.assert_allclose(x, y, atol=1e-6)


def test_ema_update_at_decay_one_keeps_ema(model: Transformer) -> None:
    """With decay=1, ema ignores ``model`` entirely."""
    perturbed = jax.tree.map(lambda x: x + 1.0 if eqx.is_inexact_array(x) else x, model)

    updated = ema_update(model, perturbed, decay=1.0)

    for x, y in zip(inexact_leaves(model), inexact_leaves(updated), strict=True):
        np.testing.assert_allclose(x, y, atol=1e-6)


def test_ema_update_linear_interpolation(model: Transformer) -> None:
    """Spot-check the Polyak formula on a concrete leaf."""
    perturbed = jax.tree.map(lambda x: x + 4.0 if eqx.is_inexact_array(x) else x, model)

    updated = ema_update(model, perturbed, decay=0.75)
    # ema_new = 0.75 * ema_old + 0.25 * model_new = 0.75 * x + 0.25 * (x + 4) = x + 1
    for before, after in zip(
        inexact_leaves(model), inexact_leaves(updated), strict=True
    ):
        np.testing.assert_allclose(after, before + 1.0, atol=1e-5)


def test_train_step_decreases_loss_on_fixed_batch(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """A few aggressive-LR steps on the same batch must lower the loss.

    Uses a tiny model + repeated batch so learning is unambiguous even
    for the stochastic MDLM objective (low-discrepancy time sampler
    averages out most noise).
    """
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)
    ema_model = clone_state(model)

    train_cfg = TrainConfig(
        learning_rate=3e-3,
        warmup_steps=5,
        max_steps=100,
        weight_decay=0.0,
        grad_clip=1.0,
        ema_decay=0.99,
    )
    optimizer, _ = make_optimizer(train_cfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    mask_id = small_config.vocab_size - 1
    train_step = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=mask_id,
        ema_decay=train_cfg.ema_decay,
    )

    batch = jnp.tile(
        jnp.arange(small_config.max_seq_len, dtype=jnp.int32)
        % (small_config.vocab_size - 1),
        (4, 1),
    )

    losses: list[float] = []
    for _ in range(50):
        key, step_key = jax.random.split(key)
        model, ema_model, opt_state, metrics, _ = train_step(
            model, ema_model, opt_state, batch, step_key
        )
        losses.append(float(metrics.loss))

    early = float(np.mean(losses[:5]))
    late = float(np.mean(losses[-5:]))
    assert late < early, f"loss did not decrease: {early:.3f} -> {late:.3f}"


def test_train_step_updates_model_and_ema(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """One step should move the model *and* produce an EMA distinct from it."""
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)
    ema_model = clone_state(model)

    train_cfg = TrainConfig(
        learning_rate=1e-2,
        warmup_steps=0,
        max_steps=10,
        ema_decay=0.5,
    )
    optimizer, _ = make_optimizer(train_cfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_step = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        ema_decay=train_cfg.ema_decay,
    )

    batch = jnp.zeros((2, small_config.max_seq_len), dtype=jnp.int32)
    key, step_key = jax.random.split(key)
    # Donate clones so the original ``model`` stays live for the
    # post-step comparisons below.
    new_model, new_ema, _new_opt_state, _metrics, _ = train_step(
        clone_state(model), ema_model, opt_state, batch, step_key
    )

    assert _count_inexact_leaves_that_changed(model, new_model) > 0
    # With decay=0.5 and the EMA starting equal to the model, the new EMA
    # must be halfway between old and new params, so it differs from both.
    assert _count_inexact_leaves_that_changed(new_model, new_ema) > 0
    assert _count_inexact_leaves_that_changed(model, new_ema) > 0


def test_train_step_jits_and_is_deterministic(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """Same key + same batch + same init produce bitwise-identical updates."""
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)

    optimizer, _ = make_optimizer(TrainConfig(warmup_steps=2, max_steps=10))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_step = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        ema_decay=0.9,
    )

    batch = jnp.zeros((2, small_config.max_seq_len), dtype=jnp.int32)

    m1, e1, _o1, mx1, _ = train_step(
        clone_state(model),
        clone_state(model),
        clone_state(opt_state),
        batch,
        jax.random.PRNGKey(123),
    )
    m2, e2, _o2, mx2, _ = train_step(
        clone_state(model),
        clone_state(model),
        clone_state(opt_state),
        batch,
        jax.random.PRNGKey(123),
    )

    assert float(mx1.loss) == pytest.approx(float(mx2.loss), abs=0.0)
    for a, b in zip(inexact_leaves(m1), inexact_leaves(m2), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(inexact_leaves(e1), inexact_leaves(e2), strict=True):
        np.testing.assert_array_equal(a, b)


def test_train_step_produces_finite_updates(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """Sanity: no nan/inf in the post-update model or EMA."""
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)

    optimizer, _ = make_optimizer(TrainConfig(warmup_steps=2, max_steps=10))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_step = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        ema_decay=0.9,
    )

    batch = jax.random.randint(
        key, (2, small_config.max_seq_len), 0, small_config.vocab_size - 1
    )
    key, step_key = jax.random.split(key)
    new_model, new_ema, _new_opt_state, metrics, _ = train_step(
        model, clone_state(model), opt_state, batch, step_key
    )

    assert jnp.isfinite(metrics.loss)
    assert jnp.isfinite(metrics.grad_norm)
    assert jnp.isfinite(metrics.param_norm)
    for tree in (new_model, new_ema):
        for leaf in inexact_leaves(tree):
            assert jnp.all(jnp.isfinite(leaf))


def test_train_step_signature_accepts_optax_chain() -> None:
    """Regression: closure capture of a chained optax optimizer traces cleanly."""
    cfg = ModelConfig(
        vocab_size=32,
        num_layers=1,
        hidden_dim=16,
        num_heads=2,
        max_seq_len=8,
    )
    model = Transformer(cfg, key=jax.random.PRNGKey(0))
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(1e-3),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    step = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=cfg.vocab_size - 1,
        ema_decay=0.9,
    )
    batch = jnp.zeros((2, cfg.max_seq_len), dtype=jnp.int32)
    _m, _e, _o, metrics, _ = step(
        model, clone_state(model), opt_state, batch, jax.random.PRNGKey(1)
    )
    assert jnp.isfinite(metrics.loss)


def test_make_train_step_narrows_via_trainstepfn_annotation(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """Generic narrowing: a ``TrainStepFn[Transformer]`` target annotation
    pins ``M = Transformer`` on the returned step, so each of its four
    return positions narrows to the concrete model type at call sites.
    """
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)

    optimizer, _ = make_optimizer(
        TrainConfig(warmup_steps=2, max_steps=10, ema_decay=0.9)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_step: TrainStepFn[Transformer] = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        ema_decay=0.9,
    )

    batch = jnp.zeros((2, small_config.max_seq_len), dtype=jnp.int32)
    key, step_key = jax.random.split(key)
    new_model, new_ema, _new_opt_state, metrics, _ = train_step(
        model, clone_state(model), opt_state, batch, step_key
    )

    assert_type(new_model, Transformer)
    assert_type(new_ema, Transformer)
    assert type(new_model) is Transformer
    assert type(new_ema) is Transformer
    assert jnp.isfinite(metrics.loss)


def test_train_step_returns_updated_key(
    key: jax.Array, small_config: ModelConfig
) -> None:
    """train_step must return a new key different from the input."""
    model = Transformer(small_config, key=key)
    optimizer, _ = make_optimizer(
        TrainConfig(warmup_steps=2, max_steps=10, ema_decay=0.9)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        ema_decay=0.9,
    )
    batch = jnp.zeros((2, small_config.max_seq_len), dtype=jnp.int32)
    input_key = jax.random.PRNGKey(42)
    expected_next = jax.random.split(input_key)[0]
    _, _, _, _, output_key = train_step(
        model, clone_state(model), opt_state, batch, input_key
    )
    # Key must have advanced (split happened inside JIT)
    assert jnp.array_equal(output_key, expected_next)


def test_prepare_batch_uses_async_device_put(small_config: ModelConfig) -> None:
    """_prepare_batch must return a sharded JAX array, not a numpy array."""
    import numpy as np  # noqa: PLC0415

    from nanodiffusion.data.cursors import PretrainCursor  # noqa: PLC0415
    from nanodiffusion.data.loader import BatchOutput  # noqa: PLC0415
    from nanodiffusion.pretrain.train import _prepare_batch  # noqa: PLC0415
    from nanodiffusion.sharding import setup_mesh  # noqa: PLC0415

    mesh = setup_mesh()
    batch = BatchOutput(
        tokens=np.zeros((4, small_config.max_seq_len), dtype=np.int32),
        segments=np.zeros((4, small_config.max_seq_len), dtype=np.int32),
        state=PretrainCursor(
            epoch=1, shard_idx=0, row_group_idx=0, doc_idx=0, token_offset=0
        ),
    )
    tokens, cursor, _stats = _prepare_batch(batch, mesh)
    assert isinstance(tokens, jax.Array)
    assert tokens.shape == (4, small_config.max_seq_len)
    assert cursor == batch.state


def test_train_config_rejects_non_positive_save_every() -> None:
    """``save_every == 0`` would ZeroDivisionError in the loop; reject up front."""
    with pytest.raises(ValidationError, match="save_every"):
        TrainConfig(save_every=0)
    with pytest.raises(ValidationError, match="save_every"):
        TrainConfig(save_every=-1)


def test_train_config_rejects_non_positive_log_every() -> None:
    with pytest.raises(ValidationError, match="log_every"):
        TrainConfig(log_every=0)


def test_train_config_rejects_ema_decay_out_of_range() -> None:
    """EMA math only makes sense for ``decay in [0, 1]``."""
    with pytest.raises(ValidationError, match="ema_decay"):
        TrainConfig(ema_decay=-0.1)
    with pytest.raises(ValidationError, match="ema_decay"):
        TrainConfig(ema_decay=1.5)


def test_train_config_accepts_ema_decay_endpoints() -> None:
    """0 and 1 are valid degenerate cases (pure model / pure EMA)."""
    TrainConfig(ema_decay=0.0)
    TrainConfig(ema_decay=1.0)
