"""Paradigm-agnostic optimizer construction + Polyak EMA shared by pretrain and SFT."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool

from nanodiffusion.config import OptimizerHyperparams


def make_optimizer(
    hp: OptimizerHyperparams,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Warmup + cosine-decay AdamW with global-norm grad clipping.

    The schedule is returned alongside the optimizer so callers can log
    the current learning rate without reaching into ``opt_state``.
    """
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=hp.learning_rate,
        warmup_steps=hp.warmup_steps,
        decay_steps=hp.max_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(hp.grad_clip),
        optax.adamw(lr_schedule, weight_decay=hp.weight_decay),
    )
    return optimizer, lr_schedule


def scale_ema_decay(decay: float, kappa: int) -> float:
    """Rescale ``decay`` for an effective batch that is ``kappa``x larger.

    Busbridge et al., *How to Scale Your EMA* (NeurIPS 2023) show that
    when the global batch grows by a factor of ``kappa``, the Polyak
    decay must shrink as ``decay ** kappa`` to keep the same effective
    averaging window in sample-space. Pure data-parallel runs use
    ``kappa = num_devices`` because each step processes ``kappa`` times
    more samples than the single-device baseline that the configured
    decay was tuned for.
    """
    return float(decay**kappa)


def ema_update[M: eqx.Module](ema_model: M, model: M, decay: float) -> M:
    """Polyak EMA on the float leaves only.

    ``ema_new = decay * ema_old + (1 - decay) * model``. Non-inexact
    leaves (ints, static fields, strings) are left untouched so integer
    bookkeeping arrays are never silently cast to float.
    """
    ema_arrays, static = eqx.partition(ema_model, eqx.is_inexact_array)
    model_arrays, _ = eqx.partition(model, eqx.is_inexact_array)
    new_ema_arrays = jax.tree.map(
        lambda e, m: decay * e + (1.0 - decay) * m, ema_arrays, model_arrays
    )
    return eqx.combine(new_ema_arrays, static)


def apply_or_skip[M: eqx.Module](
    finite: Bool[Array, ""],
    *,
    optimizer: optax.GradientTransformation,
    model: M,
    ema_model: M,
    opt_state: optax.OptState,
    grads: M,
    ema_decay: float,
) -> tuple[M, M, optax.OptState]:
    """Apply one optimizer + EMA step, or keep state unchanged when not finite.

    ``finite`` is a scalar boolean tracer: when false, the returned
    triple is the input ``(model, ema_model, opt_state)`` verbatim.
    This lets the training loop recover from transient numerical
    spikes (common at small diffusion timesteps) without aborting the
    whole run. Both branches are traced by :func:`jax.lax.cond` and the
    runtime picks one, so the skipped path still has the same HLO cost
    model.
    """

    def _apply() -> tuple[M, M, optax.OptState]:
        updates, new_opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_inexact_array),  # pyright: ignore[reportArgumentType]
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        new_model = eqx.apply_updates(model, updates)
        new_ema_model = ema_update(ema_model, new_model, ema_decay)
        return new_model, new_ema_model, new_opt_state

    def _skip() -> tuple[M, M, optax.OptState]:
        return model, ema_model, opt_state

    return jax.lax.cond(jnp.asarray(finite), _apply, _skip)
