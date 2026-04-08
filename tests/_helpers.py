"""Shared test helpers. Fixtures live in ``conftest.py``."""

from pathlib import Path

import equinox as eqx
import jax
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
