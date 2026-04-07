"""Text sources for the pretraining pipeline.

Offline-pure: only reads files already on disk. All network and
dataset-specific knowledge lives in :mod:`nanodiffusion.data.datasets`.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol, TypedDict, runtime_checkable

import pyarrow.parquet as pq

type Split = Literal["train", "val"]


class SourcePosition(TypedDict):
    """Cursor for fast-forward resume.

    ``row_group_idx`` is an opaque per-implementation cursor (parquet
    row-group index for :class:`ParquetTextSource`, batch index for
    :class:`InMemoryTextSource`).
    """

    epoch: int
    shard_idx: int
    row_group_idx: int


@runtime_checkable
class TextSource(Protocol):
    """Iterable of ``(batch, position)`` tuples with infinite iteration.

    After exhausting the assigned shards, implementations must loop back
    and increment ``position["epoch"]``. ``start`` / ``step`` partition
    the work at the implementation's natural granularity and are safe to
    use as a data-parallel sharding hook.
    """

    def iter_documents(
        self,
        split: Split,
        *,
        start: int = 0,
        step: int = 1,
        batch_size: int = 128,
    ) -> Iterator[tuple[list[str], SourcePosition]]: ...


class ParquetTextSource:
    """Streams documents from parquet shards. Strides at the row-group level."""

    def __init__(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        *,
        text_column: str = "text",
    ) -> None:
        self._train_paths = list(train_paths)
        self._val_paths = list(val_paths)
        self._text_column = text_column

    def iter_documents(
        self,
        split: Split,
        *,
        start: int = 0,
        step: int = 1,
        batch_size: int = 128,
    ) -> Iterator[tuple[list[str], SourcePosition]]:
        paths = self._train_paths if split == "train" else self._val_paths
        if not paths:
            msg = f"ParquetTextSource has no {split!r} shards"
            raise ValueError(msg)

        epoch = 1
        while True:
            for shard_idx, path in enumerate(paths):
                pf = pq.ParquetFile(path)
                for row_group_idx in range(start, pf.num_row_groups, step):
                    row_group = pf.read_row_group(
                        row_group_idx, columns=[self._text_column]
                    )
                    column = row_group.column(self._text_column).to_pylist()
                    # Parquet may contain null rows; drop them and cast.
                    docs: list[str] = [v for v in column if isinstance(v, str)]
                    for offset in range(0, len(docs), batch_size):
                        batch = docs[offset : offset + batch_size]
                        position: SourcePosition = {
                            "epoch": epoch,
                            "shard_idx": shard_idx,
                            "row_group_idx": row_group_idx,
                        }
                        yield batch, position
            epoch += 1


class InMemoryTextSource:
    """Test double: serves a fixed list of documents; last ``val_size`` are val.

    Strides at the batch level so ``row_group_idx`` matches
    :class:`ParquetTextSource`'s cursor semantics.
    """

    def __init__(self, docs: list[str], *, val_size: int = 1) -> None:
        if val_size < 1 or val_size >= len(docs):
            msg = (
                f"val_size must be in [1, {len(docs)}); got {val_size} "
                f"with {len(docs)} docs"
            )
            raise ValueError(msg)
        self._train = list(docs[:-val_size])
        self._val = list(docs[-val_size:])

    def iter_documents(
        self,
        split: Split,
        *,
        start: int = 0,
        step: int = 1,
        batch_size: int = 128,
    ) -> Iterator[tuple[list[str], SourcePosition]]:
        docs = self._train if split == "train" else self._val
        if not docs:
            msg = f"InMemoryTextSource has no {split!r} docs"
            raise ValueError(msg)

        batch_count = (len(docs) + batch_size - 1) // batch_size
        epoch = 1
        while True:
            for batch_idx in range(start, batch_count, step):
                offset = batch_idx * batch_size
                batch = docs[offset : offset + batch_size]
                position: SourcePosition = {
                    "epoch": epoch,
                    "shard_idx": 0,
                    "row_group_idx": batch_idx,
                }
                yield batch, position
            epoch += 1
