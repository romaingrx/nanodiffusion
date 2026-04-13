"""Shared train-step builder for diffusion training paths."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from nanodiffusion.metrics import CoreStepMetrics
from nanodiffusion.model import DiffusionModel
from nanodiffusion.optimizer import apply_or_skip
from nanodiffusion.types import PRNGKeyArray, Scalar

type LossFn[M: DiffusionModel, B] = Callable[[M, B, PRNGKeyArray], Scalar]

type TrainStepFn[M: DiffusionModel, B] = Callable[
    [M, M, optax.OptState, B, PRNGKeyArray],
    tuple[M, M, optax.OptState, CoreStepMetrics, PRNGKeyArray],
]


def make_train_step[M: DiffusionModel, B](
    optimizer: optax.GradientTransformation,
    *,
    loss_fn: LossFn[M, B],
    ema_decay: float,
) -> TrainStepFn[M, B]:
    """Build a JIT'd optimizer step shared by pretrain and SFT."""

    @eqx.filter_jit(donate="all")
    def train_step(
        model: M,
        ema_model: M,
        opt_state: optax.OptState,
        batch: B,
        key: PRNGKeyArray,
    ) -> tuple[M, M, optax.OptState, CoreStepMetrics, PRNGKeyArray]:
        key, step_key = jax.random.split(key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, step_key)
        grad_norm = optax.tree.norm(grads)
        finite = jnp.isfinite(grad_norm) & jnp.isfinite(loss)
        new_model, new_ema_model, new_opt_state = apply_or_skip(
            finite,
            optimizer=optimizer,
            model=model,
            ema_model=ema_model,
            opt_state=opt_state,
            grads=grads,
            ema_decay=ema_decay,
        )
        param_norm = optax.tree.norm(eqx.filter(new_model, eqx.is_inexact_array))
        metrics = CoreStepMetrics(
            loss=loss,
            grad_norm=grad_norm,
            param_norm=param_norm,
            grad_finite=finite.astype(jnp.float32),
        )
        return new_model, new_ema_model, new_opt_state, metrics, key

    return train_step
