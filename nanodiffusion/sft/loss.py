"""SFT NELBO with role-aware masking.

Same continuous-time objective as :mod:`nanodiffusion.pretrain.loss` with
two changes: (1) the forward-noise mask is intersected with a per-position
``loss_mask`` so prompt tokens stay clean in ``xt``, and (2) the batch-level
reduction is a global masked-token mean instead of a per-row mean. Built on
the per-token primitive :func:`nanodiffusion.loss.token_nll`.
"""

import jax
import jax.numpy as jnp

from nanodiffusion.loss import TimeSampler, low_discrepancy_sampler, token_nll
from nanodiffusion.model import DiffusionModel
from nanodiffusion.schedule import NoiseSchedule, loss_weight, mask_chance
from nanodiffusion.types import (
    Mask,
    MaskBatch,
    PRNGKeyArray,
    Scalar,
    TokenBatch,
    Tokens,
)


def sft_forward_mask(
    x0: Tokens,
    loss_mask: Mask,
    t: Scalar,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
) -> tuple[Tokens, Mask]:
    """Forward diffusion restricted to supervised positions.

    Identical to :func:`nanodiffusion.pretrain.loss.forward_mask` except
    the independent-coin mask is intersected with ``loss_mask``. Prompt
    positions (``loss_mask=False``) stay clean in ``xt`` so the model
    sees them as context, and unmasked response positions likewise pass
    through untouched.
    """
    chance = mask_chance(schedule, t)
    noise = jax.random.uniform(key, x0.shape)
    is_masked = (noise < chance) & loss_mask
    xt = jnp.where(is_masked, mask_token_id, x0)
    return xt, is_masked


def compute_sft_loss(
    model: DiffusionModel,
    x0: TokenBatch,
    loss_mask: MaskBatch,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
    sampler: TimeSampler = low_discrepancy_sampler,
) -> Scalar:
    """Global-masked-token NELBO for SFT.

    Per-row, applies the time-dependent weight ``w(t_i)`` to the sum of
    per-token NLL at supervised masked positions and counts how many such
    positions exist. Across the batch, the final loss is
    ``sum(weighted_nll) / max(sum(supervised_masked), 1)``. This weights
    every supervised masked token equally within the batch, instead of
    normalizing per-row first as
    :func:`nanodiffusion.pretrain.loss.compute_loss` does; the two
    aggregations diverge when rows have very different supervised token
    counts (common for SFT with padded conversations).
    """
    batch_size = x0.shape[0]
    key, t_key = jax.random.split(key)
    t_batch = sampler(batch_size, key=t_key)
    keys = jax.random.split(key, batch_size)

    def _per_sample(
        xi: Tokens,
        li: Mask,
        ti: Scalar,
        ki: PRNGKeyArray,
    ) -> tuple[Scalar, Scalar]:
        xt, is_masked = sft_forward_mask(
            xi, li, ti, schedule=schedule, mask_token_id=mask_token_id, key=ki
        )
        logits = model(xt, ti)
        weighted = loss_weight(schedule, ti) * (token_nll(logits, xi) * is_masked).sum()
        return weighted, is_masked.sum().astype(weighted.dtype)

    weighted_nll, counts = jax.vmap(_per_sample)(x0, loss_mask, t_batch, keys)
    denom = jnp.maximum(counts.sum(), 1.0)
    return weighted_nll.sum() / denom
