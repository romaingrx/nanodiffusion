import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from nanodiffusion.types import Mask, PRNGKeyArray, Scalar, Tokens


class NoiseSchedule(eqx.Module):
    """Maps continuous time t in [0, 1] to cumulative noise sigma(t)."""

    eps: float = eqx.field(static=True, default=1e-3)

    @abc.abstractmethod
    def sigma(self, t: Scalar) -> Scalar: ...

    @abc.abstractmethod
    def dsigma(self, t: Scalar) -> Scalar: ...

    def alpha(self, t: Scalar) -> Scalar:
        return jnp.exp(-self.sigma(t))

    def mask_chance(self, t: Scalar) -> Scalar:
        return -jnp.expm1(-self.sigma(t))

    def loss_weight(self, t: Scalar) -> Scalar:
        return self.dsigma(t) / jnp.expm1(self.sigma(t))


class LogLinearSchedule(NoiseSchedule):
    """sigma(t) = -log(1 - (1-eps)t), so mask_chance ~ t. Default in MDLM."""

    def sigma(self, t: Scalar) -> Scalar:
        return -jnp.log1p(-(1 - self.eps) * t)

    def dsigma(self, t: Scalar) -> Scalar:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)


class CosineSchedule(NoiseSchedule):
    """alpha(t) = eps + (1-eps)cos(pi*t/2). Slower masking at start and end."""

    def sigma(self, t: Scalar) -> Scalar:
        return -jnp.log(self.eps + (1 - self.eps) * jnp.cos(jnp.pi * t / 2))

    def dsigma(self, t: Scalar) -> Scalar:
        num = jnp.pi / 2 * (1 - self.eps) * jnp.sin(jnp.pi * t / 2)
        den = self.eps + (1 - self.eps) * jnp.cos(jnp.pi * t / 2)
        return num / den


def forward_mask(
    x0: Tokens,
    t: Scalar,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
) -> tuple[Tokens, Mask]:
    """Apply forward diffusion: independently mask each token with prob 1 - alpha(t)."""
    chance = schedule.mask_chance(t)
    noise = jax.random.uniform(key, x0.shape)
    is_masked = noise < chance
    xt = jnp.where(is_masked, mask_token_id, x0)
    return xt, is_masked
