from typing import Protocol

import jax
import jax.numpy as jnp

from nanodiffusion.model import DiffusionModel
from nanodiffusion.schedule import NoiseSchedule, loss_weight, mask_chance
from nanodiffusion.types import Logits, Mask, PRNGKeyArray, Scalar, TokenBatch, Tokens


class TimeSampler(Protocol):
    def __call__(self, batch_size: int, *, key: PRNGKeyArray) -> jax.Array: ...


def low_discrepancy_sampler(batch_size: int, *, key: PRNGKeyArray) -> jax.Array:
    """Stratified time sampling with a single random offset. See MDLM Sec. 3.3."""
    sampling_eps = 1e-5
    u = jax.random.uniform(key, (1,))
    t_batch = (u / batch_size + jnp.arange(batch_size) / batch_size) % 1
    return (1 - sampling_eps) * t_batch + sampling_eps


def forward_mask(
    x0: Tokens,
    t: Scalar,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
) -> tuple[Tokens, Mask]:
    """Apply forward diffusion: independently mask each token with prob 1 - alpha(t).

    Ref: MDLM Eq. 2 — per-token marginal q(x_t | x_0).
    """
    chance = mask_chance(schedule, t)
    noise = jax.random.uniform(key, x0.shape)
    is_masked = noise < chance
    xt = jnp.where(is_masked, mask_token_id, x0)
    return xt, is_masked


def masked_nll(
    logits: Logits,
    x0: Tokens,
    is_masked: Mask,
    weight: Scalar,
) -> Scalar:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_p_x0 = jnp.take_along_axis(log_probs, x0[..., None], axis=-1).squeeze(-1)
    nll = -log_p_x0 * is_masked
    n_masked = jnp.maximum(is_masked.sum(), 1)
    return weight * nll.sum() / n_masked


def diffusion_loss(
    model: DiffusionModel,
    x0: Tokens,
    t: Scalar,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
) -> Scalar:
    """Continuous-time NELBO for a single sequence. Assumes t > 0.

    w(t) * mean_masked(-log p(x0 | xt)), see MDLM Eq. 13-14.
    """
    xt, is_masked = forward_mask(
        x0, t, schedule=schedule, mask_token_id=mask_token_id, key=key
    )
    logits = model(xt, t)
    return masked_nll(logits, x0, is_masked, loss_weight(schedule, t))


def compute_loss(
    model: DiffusionModel,
    x0: TokenBatch,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
    sampler: TimeSampler = low_discrepancy_sampler,
) -> Scalar:
    batch_size = x0.shape[0]
    key, t_key = jax.random.split(key)
    t_batch = sampler(batch_size, key=t_key)

    keys = jax.random.split(key, batch_size)

    def _per_sample(xi: Tokens, ti: Scalar, ki: PRNGKeyArray) -> Scalar:
        return diffusion_loss(
            model, xi, ti, schedule=schedule, mask_token_id=mask_token_id, key=ki
        )

    losses = jax.vmap(_per_sample)(x0, t_batch, keys)
    return losses.mean()
