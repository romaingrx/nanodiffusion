import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from nanodiffusion.config import ModelConfig
from nanodiffusion.model.transformer import Transformer
from nanodiffusion.pretrain.loss import (
    compute_loss,
    diffusion_loss,
    forward_mask,
    masked_nll,
)
from nanodiffusion.schedule import (
    CosineSchedule,
    LogLinearSchedule,
    NoiseSchedule,
    alpha,
    loss_weight,
    mask_chance,
)


@pytest.fixture(params=[LogLinearSchedule, CosineSchedule])
def schedule(request: pytest.FixtureRequest) -> NoiseSchedule:
    return request.param()


def test_mask_chance_zero_at_t0(schedule: NoiseSchedule) -> None:
    assert jnp.isclose(mask_chance(schedule, jnp.array(0.0)), 0.0, atol=1e-6)


def test_mask_chance_near_one_at_t1(schedule: NoiseSchedule) -> None:
    assert mask_chance(schedule, jnp.array(1.0)) > 0.99


def test_dsigma_positive(schedule: NoiseSchedule) -> None:
    for t in jnp.linspace(0.01, 0.99, 10):
        assert schedule.dsigma(t) > 0


def test_alpha_plus_mask_chance_equals_one(schedule: NoiseSchedule) -> None:
    for t in jnp.linspace(0.0, 1.0, 20):
        total = alpha(schedule, t) + mask_chance(schedule, t)
        assert jnp.isclose(total, 1.0, atol=1e-6)


def test_loss_weight_positive(schedule: NoiseSchedule) -> None:
    for t in jnp.linspace(0.01, 0.99, 10):
        assert loss_weight(schedule, t) > 0


def test_forward_mask_nothing_at_t0(schedule: NoiseSchedule, key: jax.Array) -> None:
    x0 = jnp.arange(16)
    xt, is_masked = forward_mask(
        x0, jnp.array(0.0), schedule=schedule, mask_token_id=999, key=key
    )
    assert not jnp.any(is_masked)
    assert jnp.array_equal(xt, x0)


def test_forward_mask_almost_everything_at_t1(
    schedule: NoiseSchedule, key: jax.Array
) -> None:
    x0 = jnp.arange(64)
    _xt, is_masked = forward_mask(
        x0, jnp.array(1.0), schedule=schedule, mask_token_id=999, key=key
    )
    assert is_masked.sum() > 50


def test_forward_mask_replaces_with_mask_token(
    schedule: NoiseSchedule, key: jax.Array
) -> None:
    x0 = jnp.arange(16)
    mask_id = 999
    xt, is_masked = forward_mask(
        x0, jnp.array(0.5), schedule=schedule, mask_token_id=mask_id, key=key
    )
    assert jnp.all(xt[is_masked] == mask_id)
    assert jnp.array_equal(xt[~is_masked], x0[~is_masked])


def test_diffusion_loss_shape(
    model: Transformer,
    small_config: ModelConfig,
    schedule: NoiseSchedule,
    key: jax.Array,
) -> None:
    x0 = jax.random.randint(key, (16,), 0, small_config.vocab_size)
    loss = diffusion_loss(
        model,
        x0,
        jnp.array(0.5),
        schedule=schedule,
        mask_token_id=small_config.vocab_size - 1,
        key=key,
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert loss >= 0


def test_gradient_flows_only_through_masked(
    model: Transformer, small_config: ModelConfig, key: jax.Array
) -> None:
    k1, k2 = jax.random.split(key)
    x0 = jax.random.randint(k1, (16,), 0, small_config.vocab_size)

    sched = LogLinearSchedule()
    mask_id = small_config.vocab_size - 1
    t = jnp.array(0.5)

    xt, is_masked = forward_mask(x0, t, schedule=sched, mask_token_id=mask_id, key=k2)
    weight = loss_weight(sched, t)
    logits = model(xt, t)

    def _nll_from_logits(lg: jax.Array) -> jax.Array:
        return masked_nll(lg, x0, is_masked, weight)

    grad_logits = jax.grad(_nll_from_logits)(logits)

    assert jnp.allclose(grad_logits[~is_masked], 0.0, atol=1e-7)
    assert jnp.any(grad_logits[is_masked] != 0)


def test_diffusion_loss_gradient_flow(
    model: Transformer,
    small_config: ModelConfig,
    schedule: NoiseSchedule,
    key: jax.Array,
) -> None:
    x0 = jax.random.randint(key, (16,), 0, small_config.vocab_size)

    @eqx.filter_grad
    def compute_grad(m: Transformer) -> jax.Array:
        return diffusion_loss(
            m,
            x0,
            jnp.array(0.5),
            schedule=schedule,
            mask_token_id=small_config.vocab_size - 1,
            key=key,
        )

    grads = compute_grad(model)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)


def test_compute_loss_batched(
    model: Transformer,
    small_config: ModelConfig,
    schedule: NoiseSchedule,
    key: jax.Array,
) -> None:
    x0 = jax.random.randint(key, (4, 16), 0, small_config.vocab_size)
    loss = compute_loss(
        model,
        x0,
        schedule=schedule,
        mask_token_id=small_config.vocab_size - 1,
        key=key,
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_compute_loss_jit_compatible(
    model: Transformer,
    small_config: ModelConfig,
    schedule: NoiseSchedule,
    key: jax.Array,
) -> None:
    x0 = jax.random.randint(key, (4, 16), 0, small_config.vocab_size)
    mask_id = small_config.vocab_size - 1

    eager = compute_loss(model, x0, schedule=schedule, mask_token_id=mask_id, key=key)

    @eqx.filter_jit
    def jit_loss(m: Transformer, batch: jax.Array, k: jax.Array) -> jax.Array:
        return compute_loss(m, batch, schedule=schedule, mask_token_id=mask_id, key=k)

    jitted = jit_loss(model, x0, key)
    assert jnp.allclose(eager, jitted, atol=1e-5)


def test_compute_loss_gradient_flow(
    model: Transformer,
    small_config: ModelConfig,
    schedule: NoiseSchedule,
    key: jax.Array,
) -> None:
    x0 = jax.random.randint(key, (4, 16), 0, small_config.vocab_size)
    mask_id = small_config.vocab_size - 1

    @eqx.filter_grad
    def compute_grad(m: Transformer) -> jax.Array:
        return compute_loss(m, x0, schedule=schedule, mask_token_id=mask_id, key=key)

    grads = compute_grad(model)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)


def test_diffusion_loss_value_correctness(key: jax.Array) -> None:
    sched = LogLinearSchedule()
    t = jnp.array(0.5)
    vocab_size = 8
    mask_id = vocab_size - 1
    x0 = jnp.array([0, 1, 2, 3])

    _xt, is_masked = forward_mask(x0, t, schedule=sched, mask_token_id=mask_id, key=key)

    # Uniform logits → NLL = log(vocab_size)
    expected_nll = jnp.log(jnp.array(vocab_size, dtype=jnp.float32))
    n_masked = is_masked.sum()
    weight = loss_weight(sched, t)
    expected_loss = jnp.where(n_masked > 0, weight * expected_nll, 0.0)

    config = ModelConfig(
        vocab_size=vocab_size, num_layers=1, hidden_dim=32, num_heads=2, max_seq_len=8
    )
    model = Transformer(config, key=key)
    # Zero out lm_head to get uniform logits for a predictable expected loss
    model = eqx.tree_at(
        lambda m: m.lm_head.weight, model, jnp.zeros_like(model.lm_head.weight)
    )
    loss = diffusion_loss(model, x0, t, schedule=sched, mask_token_id=mask_id, key=key)
    if n_masked > 0:
        assert jnp.isclose(loss, expected_loss, rtol=0.1)
