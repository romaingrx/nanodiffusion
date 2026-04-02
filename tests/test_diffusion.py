import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from nanodiffusion.config import ModelConfig
from nanodiffusion.loss import compute_loss, diffusion_loss
from nanodiffusion.model.transformer import Transformer
from nanodiffusion.schedule import (
    CosineSchedule,
    LogLinearSchedule,
    NoiseSchedule,
    forward_mask,
)


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        num_layers=2,
        hidden_dim=64,
        num_heads=4,
        max_seq_len=32,
    )


@pytest.fixture
def key() -> jax.Array:
    return jax.random.PRNGKey(0)


@pytest.fixture(params=[LogLinearSchedule, CosineSchedule])
def schedule(request: pytest.FixtureRequest) -> NoiseSchedule:
    return request.param()


# -- Schedule --


def test_mask_chance_zero_at_t0(schedule: NoiseSchedule) -> None:
    assert jnp.isclose(schedule.mask_chance(jnp.array(0.0)), 0.0, atol=1e-6)


def test_mask_chance_near_one_at_t1(schedule: NoiseSchedule) -> None:
    assert schedule.mask_chance(jnp.array(1.0)) > 0.99


def test_dsigma_positive(schedule: NoiseSchedule) -> None:
    for t in jnp.linspace(0.01, 0.99, 10):
        assert schedule.dsigma(t) > 0


# -- Forward masking --


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


# -- Loss --


def test_diffusion_loss_shape(
    small_config: ModelConfig, schedule: NoiseSchedule, key: jax.Array
) -> None:
    k1, k2 = jax.random.split(key)
    model = Transformer(small_config, key=k1)
    x0 = jax.random.randint(k2, (16,), 0, small_config.vocab_size)

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
    small_config: ModelConfig, key: jax.Array
) -> None:
    k1, k2, k3 = jax.random.split(key, 3)
    model = Transformer(small_config, key=k1)
    x0 = jax.random.randint(k2, (16,), 0, small_config.vocab_size)

    schedule = LogLinearSchedule()
    mask_id = small_config.vocab_size - 1
    t = jnp.array(0.5)

    xt, is_masked = forward_mask(
        x0, t, schedule=schedule, mask_token_id=mask_id, key=k3
    )

    def loss_from_logits(logits: jax.Array) -> jax.Array:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_p_x0 = jnp.take_along_axis(log_probs, x0[..., None], axis=-1).squeeze(-1)
        nll = -log_p_x0 * is_masked
        n_masked = jnp.maximum(is_masked.sum(), 1)
        weight = schedule.loss_weight(t)
        return jnp.where(is_masked.sum() > 0, weight * nll.sum() / n_masked, 0.0)

    logits = model(xt, t)
    grad_logits = jax.grad(loss_from_logits)(logits)

    assert jnp.allclose(grad_logits[~is_masked], 0.0, atol=1e-7)
    assert jnp.any(grad_logits[is_masked] != 0)


def test_diffusion_loss_gradient_flow(
    small_config: ModelConfig, schedule: NoiseSchedule, key: jax.Array
) -> None:
    k1, k2 = jax.random.split(key)
    model = Transformer(small_config, key=k1)
    x0 = jax.random.randint(k2, (16,), 0, small_config.vocab_size)

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
    small_config: ModelConfig, schedule: NoiseSchedule, key: jax.Array
) -> None:
    k1, k2 = jax.random.split(key)
    model = Transformer(small_config, key=k1)
    x0 = jax.random.randint(k2, (4, 16), 0, small_config.vocab_size)

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
    small_config: ModelConfig, schedule: NoiseSchedule, key: jax.Array
) -> None:
    k1, k2 = jax.random.split(key)
    model = Transformer(small_config, key=k1)
    x0 = jax.random.randint(k2, (4, 16), 0, small_config.vocab_size)
    mask_id = small_config.vocab_size - 1

    eager = compute_loss(model, x0, schedule=schedule, mask_token_id=mask_id, key=key)

    @eqx.filter_jit
    def jit_loss(m: Transformer, batch: jax.Array, k: jax.Array) -> jax.Array:
        return compute_loss(
            m, batch, schedule=schedule, mask_token_id=mask_id, key=k
        )

    jitted = jit_loss(model, x0, key)
    assert jnp.allclose(eager, jitted, atol=1e-5)
