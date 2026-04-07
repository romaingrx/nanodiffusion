"""Text sources for the pretraining data pipeline.

A ``TextSource`` is a thin abstraction over an iterable of document batches.
This module is offline-pure: it only reads files that already exist on disk.
All network and dataset-specific knowledge lives in
``nanodiffusion.data.datasets``.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol, TypedDict, runtime_checkable

import pyarrow.parquet as pq

type Split = Literal["train", "val"]


class SourcePosition(TypedDict):
    """Where the source is in its iteration; used for fast-forward resume.

    ``epoch`` is 1-indexed and increments on each full pass through the
    assigned shards. ``shard_idx`` is the index into the per-split shard
    list (not a global file index). ``row_group_idx`` is the row group
    within that shard at which the most recently yielded batch began.
    """

    epoch: int
    shard_idx: int
    row_group_idx: int


@runtime_checkable
class TextSource(Protocol):
    """Iterable of document batches with resume positions."""

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
    datasets. The ``train_paths`` and ``val_paths`` lists let the caller
    decide which files belong to which split. Iteration is infinite:
    after exhausting the assigned shards, the source loops back and
    increments ``epoch``.

    ``start`` / ``step`` are exposed on ``iter_documents`` so a future
    data-parallel launcher can shard row groups across hosts without
    rewriting this class.
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
                    # Drop null rows: parquet may contain them and they are
                    # useless for pretraining. The cast is safe after filter.
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

    Reserves the last ``val_size`` documents as the val split; the rest
    are train. Iteration is infinite: incrementing ``epoch`` on each pass.
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

        # Treat the whole list as one shard with one row-group per stride.
        sliced = docs[start::step]
        epoch = 1
        while True:
            for offset in range(0, len(sliced), batch_size):
                batch = sliced[offset : offset + batch_size]
                position: SourcePosition = {
                    "epoch": epoch,
                    "shard_idx": 0,
                    "row_group_idx": offset,
                }
                yield batch, position
            epoch += 1
