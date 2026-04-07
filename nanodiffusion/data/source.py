"""Text sources for the pretraining data pipeline.

A :class:`TextSource` is a thin abstraction over an iterable of document
batches. This module is offline-pure: it only reads files that already exist
on disk. All network and dataset-specific knowledge lives in
:mod:`nanodiffusion.data.datasets`.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol, TypedDict, runtime_checkable

import pyarrow.parquet as pq

type Split = Literal["train", "val"]


class SourcePosition(TypedDict):
    """Where the source is in its iteration; used for fast-forward resume.

    ``epoch`` is 1-indexed and increments on each full pass through the
    assigned shards.

    ``shard_idx`` is the index into the per-split shard list (not a global
    file index).

    ``row_group_idx`` is the index of the current group of records within
    that shard. Both implementations use it as an opaque cursor: for
    :class:`ParquetTextSource` it is the parquet row-group index; for
    :class:`InMemoryTextSource` it is the batch index within the (single)
    in-memory shard. Polymorphic resume code should treat it as opaque.
    """

    epoch: int
    shard_idx: int
    row_group_idx: int


@runtime_checkable
class TextSource(Protocol):
    """Iterable of document batches with resume positions.

    Implementations must yield ``(batch, position)`` tuples indefinitely:
    after exhausting the assigned shards they must loop back to the start
    and increment ``position["epoch"]``. The pretrain loader assumes
    iteration never terminates; finite sources will surface as a
    ``RuntimeError`` deep inside the generator.

    The ``start`` / ``step`` parameters partition the work at the
    implementation's natural granularity (parquet row-group for parquet,
    batch for in-memory). For any fixed source and parameters, the same
    ``(start, step)`` values produce a deterministic, non-overlapping
    subset of records, which makes them safe to use as a data-parallel
    sharding hook.
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
    """Streams text documents from a list of parquet shards on disk.

    Pure offline reader. Knows nothing about Hugging Face or specific
    datasets. The caller decides which files belong to which split via
    ``train_paths`` / ``val_paths``. Iteration is infinite.

    ``start`` / ``step`` stride at the *row-group* level within each shard.
    """

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
                    # Drop null rows. Parquet may contain them and they are
                    # useless for pretraining; the cast is safe after filter.
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
    """Test double: serves a fixed list of documents.

    Reserves the last ``val_size`` documents as the val split; the rest are
    train. Iteration is infinite. ``start`` / ``step`` stride at the *batch*
    level (one row-group equals one batch in the in-memory layout), so
    ``SourcePosition.row_group_idx`` is the batch index within the single
    in-memory shard. This keeps the partition contract from
    :class:`TextSource` consistent across implementations.
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

        # Group docs into batches and stride over the batch indices so the
        # (start, step) partition contract holds with parquet semantics.
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
