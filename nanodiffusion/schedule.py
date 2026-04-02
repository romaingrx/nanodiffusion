from typing import Protocol

import equinox as eqx
import jax.numpy as jnp

from nanodiffusion.types import Scalar


class NoiseSchedule(Protocol):
    """Cumulative noise sigma(t) and its derivative for t in [0, 1].

    Ref: Sahoo et al., "Simple and Effective Masked Diffusion Language Models"
    (MDLM), NeurIPS 2024, Sec. 3.1 — sigma parameterization.
    """

    def sigma(self, t: Scalar) -> Scalar: ...
    def dsigma(self, t: Scalar) -> Scalar: ...


def alpha(schedule: NoiseSchedule, t: Scalar) -> Scalar:
    """Probability a token stays unmasked: exp(-sigma(t))."""
    return jnp.exp(-schedule.sigma(t))


def mask_chance(schedule: NoiseSchedule, t: Scalar) -> Scalar:
    """Probability a token is masked: 1 - alpha(t)."""
    return -jnp.expm1(-schedule.sigma(t))


def loss_weight(schedule: NoiseSchedule, t: Scalar) -> Scalar:
    """NELBO weight: dsigma/dt / expm1(sigma(t)). See MDLM Eq. 14."""
    return schedule.dsigma(t) / jnp.expm1(schedule.sigma(t))


class LogLinearSchedule(eqx.Module):
    """sigma(t) = -log(1 - (1-eps)t), so mask_chance ~ t.

    Default in MDLM (Sahoo et al., 2024) and LLaDA (Nie et al., 2025).
    """
    eps: float = eqx.field(static=True, default=1e-3)

    def sigma(self, t: Scalar) -> Scalar:
        return -jnp.log1p(-(1 - self.eps) * t)

    def dsigma(self, t: Scalar) -> Scalar:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)


class CosineSchedule(eqx.Module):
    """alpha(t) = eps + (1-eps)cos(pi*t/2). Slower masking at start and end.

    From Chang et al., "MaskGIT", CVPR 2022.
    """
    eps: float = eqx.field(static=True, default=1e-3)

    def sigma(self, t: Scalar) -> Scalar:
        return -jnp.log(self.eps + (1 - self.eps) * jnp.cos(jnp.pi * t / 2))

    def dsigma(self, t: Scalar) -> Scalar:
        num = jnp.pi / 2 * (1 - self.eps) * jnp.sin(jnp.pi * t / 2)
        den = self.eps + (1 - self.eps) * jnp.cos(jnp.pi * t / 2)
        return num / den
