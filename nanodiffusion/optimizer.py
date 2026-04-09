"""Paradigm-agnostic optimizer construction + Polyak EMA shared by pretrain and SFT."""

import equinox as eqx
import jax
import optax

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
