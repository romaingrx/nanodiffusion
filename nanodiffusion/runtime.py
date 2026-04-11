"""Small runtime helpers shared by training entrypoints."""

from pathlib import Path

import equinox as eqx
import optax
from jax.sharding import Mesh

from nanodiffusion.sharding import replicate
from nanodiffusion.types import PRNGKeyArray


def configure_jax_runtime(run_dir: Path) -> None:
    """Apply the per-run JAX settings shared by pretrain and SFT."""
    import jax  # noqa: PLC0415

    jax.config.update("jax_compilation_cache_dir", str(run_dir / ".jax_cache"))
    jax.config.update("jax_explain_cache_misses", True)  # noqa: FBT003
    jax.config.update("jax_optimization_level", "O1")


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
