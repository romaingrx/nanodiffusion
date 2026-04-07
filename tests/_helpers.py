"""Shared utilities for tests.

This module is intentionally separate from ``conftest.py``: pytest treats
``conftest.py`` specially during collection, and importing helpers FROM it
is non-idiomatic. Plain helper functions live here; ``conftest.py`` is
reserved for fixtures.
"""

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def take[T](it: Iterator[T], n: int) -> list[T]:
    """Take the first ``n`` items from an iterator."""
    out: list[T] = []
    for i, item in enumerate(it):
        if i >= n:
            break
        out.append(item)
    return out


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
