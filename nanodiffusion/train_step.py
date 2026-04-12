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
    tuple[M, M, optax.OptState, CoreStepMetrics],
]


def _wrap_with_accum[M: DiffusionModel, B](
    loss_fn: LossFn[M, B],
    grad_accum_steps: int,
) -> LossFn[M, B]:
    """Wrap ``loss_fn`` to scan over micro-batches and average losses.

    The backward pass through ``jax.lax.scan`` naturally accumulates
    gradients across micro-batches. ``jax.checkpoint`` on the scan body
    ensures only one micro-batch's activations are live at a time, so
    peak memory stays at O(micro_batch) regardless of ``grad_accum_steps``.

    GSPMD inserts the DP all-reduce on the final accumulated gradient,
    amortizing communication cost by ``grad_accum_steps``×.
    """

    def accumulated_loss(model: M, batch: B, key: PRNGKeyArray) -> Scalar:
        micro_batches = batch.reshape(grad_accum_steps, -1, *batch.shape[1:])
        keys = jax.random.split(key, grad_accum_steps)

        @jax.checkpoint
        def body(micro: B, k: PRNGKeyArray) -> Scalar:
            return loss_fn(model, micro, k)

        def scan_fn(_carry: None, xs: tuple[B, PRNGKeyArray]) -> tuple[None, Scalar]:
            micro, k = xs
            return None, body(micro, k)

        _, losses = jax.lax.scan(scan_fn, None, (micro_batches, keys))
        return losses.mean()

    return accumulated_loss


def make_train_step[M: DiffusionModel, B](
    optimizer: optax.GradientTransformation,
    *,
    loss_fn: LossFn[M, B],
    ema_decay: float,
    grad_accum_steps: int = 1,
) -> TrainStepFn[M, B]:
    """Build a JIT'd optimizer step shared by pretrain and SFT."""

    if grad_accum_steps > 1:
        loss_fn = _wrap_with_accum(loss_fn, grad_accum_steps)

    @eqx.filter_jit(donate="all")
    def train_step(
        model: M,
        ema_model: M,
        opt_state: optax.OptState,
        batch: B,
        key: PRNGKeyArray,
    ) -> tuple[M, M, optax.OptState, CoreStepMetrics]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, key)
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
        return new_model, new_ema_model, new_opt_state, metrics

    return train_step
