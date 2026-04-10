"""Data-parallel sharding for multi-device training.

For models that fit on a single device (up to ~1.5B params), pure data
parallelism is optimal: every device holds a full replica of the model
and processes a shard of the batch. JAX's GSPMD partitioner infers
gradient all-reduce automatically when a replicated parameter is
differentiated through a sharded computation.

The helpers here are intentionally thin wrappers around
:mod:`jax.sharding` so they stay readable and the caller (pretrain /
SFT driver) keeps full control of what gets placed where. Single-chip
runs go through the same code path: a 1-device mesh makes every
``device_put`` a no-op.
"""

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from nanodiffusion.types import PRNGKeyArray


def setup_mesh() -> Mesh:
    """Single-axis data-parallel mesh over all local devices."""
    return Mesh(jax.devices(), ("dp",))


def replicate[T](tree: T, mesh: Mesh) -> T:
    """Place every leaf of *tree* replicated on all mesh devices."""
    return jax.device_put(tree, NamedSharding(mesh, P()))


def shard_batch[T](batch: T, mesh: Mesh) -> T:
    """Shard every array leaf of *batch* along its first axis across ``dp``.

    Works with both plain arrays (pretrain) and pytrees like
    :class:`SFTJaxBatch` where every leaf is a ``(batch, seq)`` array
    that should be split the same way.
    """
    return jax.device_put(batch, NamedSharding(mesh, P("dp")))


def shard_keys(key: PRNGKeyArray, mesh: Mesh) -> jax.Array:
    """Split *key* into one sub-key per device and shard across ``dp``.

    Each device receives a unique PRNG key so per-element noise in
    :func:`jax.vmap`'d loss functions is independent across the batch
    shards. Without this, every device would sample identical diffusion
    noise and DP would collapse to a replicated single-device run.
    """
    num_devices = len(mesh.devices.flat)
    keys = jax.random.split(key, num_devices)
    return jax.device_put(keys, NamedSharding(mesh, P("dp")))
