"""Small runtime helpers shared by training entrypoints."""

from pathlib import Path

import equinox as eqx
import optax
from jax.sharding import Mesh

from nanodiffusion.sharding import replicate
from nanodiffusion.types import PRNGKeyArray


def configure_jax_runtime(run_dir: Path) -> None:
    """Apply the per-run JAX settings shared by pretrain and SFT.

    We deliberately *don't* override ``jax_optimization_level`` — the
    JAX default is "O2" on TPU which enables full XLA optimization
    passes (layout, fusion, scheduling). Overriding to "O1" was
    measured to drop steady-state tok/s by ~30% on v6e-4 because O1
    means "faster compile, less runtime optimization", not "latency
    hiding collectives" as the name might suggest.
    """
    import jax  # noqa: PLC0415

    jax.config.update("jax_compilation_cache_dir", str(run_dir / ".jax_cache"))
    jax.config.update("jax_explain_cache_misses", True)  # noqa: FBT003


def place_training_state[M: eqx.Module](
    model: M,
    ema_model: M,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> tuple[M, M, optax.OptState, PRNGKeyArray]:
    """Replicate the training state over the chosen mesh."""
    return (
        replicate(model, mesh),
        replicate(ema_model, mesh),
        replicate(opt_state, mesh),
        replicate(key, mesh),
    )
