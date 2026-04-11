"""Shared test helpers. Fixtures live in ``conftest.py``."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(
    path: Path,
    texts: list[str | None],
    *,
    row_group_size: int = 2,
    column: str = "text",
) -> None:
    """Write a tiny parquet file for tests. ``None`` entries become null rows."""
    table = pa.table({column: texts})
    pq.write_table(table, path, row_group_size=row_group_size)


def inexact_leaves(tree: eqx.Module) -> list[jax.Array]:
    """Return the float-array leaves of an equinox module."""
    return jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))


def clone_state[T](tree: T) -> T:
    """Deep-copy the array leaves of a pytree, preserving structure.

    The JIT'd train step uses ``donate="all"``, which (a) errors out
    when two of its inputs alias the same buffers ("donate the same
    buffer twice") and (b) invalidates the buffers after the call so a
    second call with the same inputs would fail. Tests that construct
    their own state (skipping the fresh-state loader) use this to
    produce independent input buffers — either to stand in for a
    freshly-initialised EMA, or to re-run the same step twice for a
    determinism check.
    """
    return jax.tree.map(lambda x: jnp.copy(x) if eqx.is_array(x) else x, tree)
