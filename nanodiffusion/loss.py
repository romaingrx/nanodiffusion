import jax
import jax.numpy as jnp

from nanodiffusion.model.transformer import Transformer
from nanodiffusion.schedule import NoiseSchedule, forward_mask
from nanodiffusion.types import PRNGKeyArray, Scalar, TokenBatch, Tokens


def diffusion_loss(
    model: Transformer,
    x0: Tokens,
    t: Scalar,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
) -> Scalar:
    """Weighted cross-entropy on masked positions for a single sequence.

    Implements the continuous-time NELBO: w(t) * mean_masked(-log p(x0 | xt)),
    where w(t) = dsigma/dt / expm1(sigma(t)).
    """
    xt, is_masked = forward_mask(
        x0, t, schedule=schedule, mask_token_id=mask_token_id, key=key
    )
    logits = model(xt, t)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_p_x0 = jnp.take_along_axis(log_probs, x0[..., None], axis=-1).squeeze(-1)

    nll = -log_p_x0 * is_masked
    n_masked = jnp.maximum(is_masked.sum(), 1)
    mean_nll = nll.sum() / n_masked
    weight = schedule.loss_weight(t)
    return jnp.where(is_masked.sum() > 0, weight * mean_nll, 0.0)


def compute_loss(
    model: Transformer,
    x0: TokenBatch,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
) -> Scalar:
    """Batched diffusion loss with low-discrepancy time sampling."""
    batch_size = x0.shape[0]
    key, t_key = jax.random.split(key)

    eps = 1e-5
    u = jax.random.uniform(t_key, (1,))
    t_batch = (u / batch_size + jnp.arange(batch_size) / batch_size) % 1
    t_batch = (1 - eps) * t_batch + eps

    keys = jax.random.split(key, batch_size)

    def _per_sample(xi: Tokens, ti: Scalar, ki: PRNGKeyArray) -> Scalar:
        return diffusion_loss(
            model, xi, ti, schedule=schedule, mask_token_id=mask_token_id, key=ki
        )

    losses = jax.vmap(_per_sample)(x0, t_batch, keys)
    return losses.mean()
