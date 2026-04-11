"""Data-parallel sharding for multi-device training.

For models that fit on a single device (up to ~1.5B params), pure data
parallelism is optimal: every device holds a full replica of the model
and processes a shard of the batch. JAX's GSPMD partitioner infers
gradient all-reduce automatically when a replicated parameter is
differentiated through a sharded computation.

The helpers here are intentionally thin wrappers around
:mod:`jax.sharding` so they stay readable and the caller (pretrain /
SFT driver) keeps full control of what gets placed where. Single-chip
runs go through the same code path: a 1x1 mesh makes every
``device_put`` a no-op.
"""

import math

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from nanodiffusion.types import PRNGKeyArray

DP_AXES: tuple[str, str] = ("X", "Y")
"""2D data-parallel mesh axis names. Batches shard over ``P(DP_AXES)``
so the first dim is split across both ICI dimensions concurrently."""


def _largest_factor_pair(n: int) -> tuple[int, int]:
    """Return ``(a, b)`` with ``a * b == n`` and ``a >= b >= 1``,
    maximizing ``b`` (i.e. the squarest possible rectangle)."""
    for b in range(math.isqrt(n), 0, -1):
        if n % b == 0:
            return n // b, b
    return n, 1  # unreachable for n >= 1


def setup_mesh() -> Mesh:
    """2D data-parallel mesh over all local devices.

    The device array is reshaped into the squarest ``(X, Y)`` rectangle
    that fits the device count; batches are then sharded over
    ``P(("X", "Y"))`` so the first dim splits across both ICI
    dimensions. On v5e-8 this yields a 4x2 mesh, doubling the
    effective ICI bandwidth vs a single-axis DP mesh (the scaling
    book's ``B/X > C/(W_ici * M)`` threshold drops from ~2550 to
    ~1275 tokens per chip). Single-chip runs collapse to a 1x1 mesh
    and behave identically to the prior single-axis layout.
    """
    devices = np.asarray(jax.devices())
    x, y = _largest_factor_pair(len(devices))
    return Mesh(devices.reshape(x, y), DP_AXES)


def replicate[T](tree: T, mesh: Mesh) -> T:
    """Place every leaf of *tree* replicated on all mesh devices."""
    return jax.device_put(tree, NamedSharding(mesh, P()))


def shard_batch[T](batch: T, mesh: Mesh) -> T:
    """Shard every array leaf of *batch* along its first axis across
    the 2D ``(X, Y)`` data-parallel axes.

    Works with both plain arrays (pretrain) and pytrees like
    :class:`SFTJaxBatch` where every leaf is a ``(batch, seq)`` array
    that should be split the same way. ``P(("X", "Y"))`` flattens the
    split across both mesh axes so the batch dimension is distributed
    over all devices even on a rectangular topology.
    """
    return jax.device_put(batch, NamedSharding(mesh, P(DP_AXES)))


def shard_keys(key: PRNGKeyArray, mesh: Mesh) -> jax.Array:
    """Split *key* into one sub-key per device and shard across
    ``P(("X", "Y"))``.

    Each device receives a unique PRNG key so per-element noise in
    :func:`jax.vmap`'d loss functions is independent across the batch
    shards. Without this, every device would sample identical diffusion
    noise and DP would collapse to a replicated single-device run.
    """
    num_devices = len(mesh.devices.flat)
    keys = jax.random.split(key, num_devices)
    return jax.device_put(keys, NamedSharding(mesh, P(DP_AXES)))
