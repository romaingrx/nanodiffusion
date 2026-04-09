"""Text sources for the pretraining pipeline.

Offline-pure: only reads files already on disk. All network and
dataset-specific knowledge lives in :mod:`nanodiffusion.data.datasets`.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import pyarrow.parquet as pq

from nanodiffusion.data.cursors import PretrainCursor

type Split = Literal["train", "val"]


@runtime_checkable
class TextSource(Protocol):
    """Iterable of ``(batch, position)`` tuples with infinite iteration.

    After exhausting the assigned shards, implementations must loop back
    and increment ``position.epoch``. ``start`` / ``step`` partition the
    work at the implementation's natural granularity and are safe to use
    as a data-parallel sharding hook. ``resume`` fast-forwards past a
    previously yielded position so checkpoint resumption skips already
    processed data.
    """

    def iter_documents(
        self,
        split: Split,
        *,
        start: int = 0,
        step: int = 1,
        batch_size: int = 128,
        resume: PretrainCursor | None = None,
    ) -> Iterator[tuple[list[str], PretrainCursor]]: ...


def _validate_stride(start: int, step: int) -> None:
    if start < 0 or step < 1:
        msg = (
            f"start must be >= 0 and step must be >= 1, got start={start}, step={step}"
        )
        raise ValueError(msg)


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
        resume: PretrainCursor | None = None,
    ) -> Iterator[tuple[list[str], PretrainCursor]]:
        paths = self._train_paths if split == "train" else self._val_paths
        if not paths:
            msg = f"ParquetTextSource has no {split!r} shards"
            raise ValueError(msg)
        _validate_stride(start, step)

        epoch = resume.epoch if resume else 1
        skip = resume
        while True:
            yielded = False
            for shard_idx, path in enumerate(paths):
                if skip is not None and shard_idx < skip.shard_idx:
                    continue
                pf = pq.ParquetFile(path)
                if skip is not None and shard_idx == skip.shard_idx:
                    rg_start = skip.row_group_idx + step
                else:
                    rg_start = start
                for row_group_idx in range(rg_start, pf.num_row_groups, step):
                    row_group = pf.read_row_group(
                        row_group_idx, columns=[self._text_column]
                    )
                    column = row_group.column(self._text_column).to_pylist()
                    # Parquet may contain null rows; drop them and cast.
                    docs: list[str] = [v for v in column if isinstance(v, str)]
                    for offset in range(0, len(docs), batch_size):
                        batch = docs[offset : offset + batch_size]
                        position = PretrainCursor(
                            epoch=epoch,
                            shard_idx=shard_idx,
                            row_group_idx=row_group_idx,
                        )
                        yielded = True
                        yield batch, position
            was_skipping = skip is not None
            skip = None
            if not yielded and not was_skipping:
                # An unskipped epoch produced nothing, so the next will too.
                msg = (
                    f"ParquetTextSource yields no batches for split={split!r} "
                    f"with start={start}, step={step}"
                )
                raise ValueError(msg)
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
        resume: PretrainCursor | None = None,
    ) -> Iterator[tuple[list[str], PretrainCursor]]:
        docs = self._train if split == "train" else self._val
        if not docs:
            msg = f"InMemoryTextSource has no {split!r} docs"
            raise ValueError(msg)
        _validate_stride(start, step)

        batch_count = (len(docs) + batch_size - 1) // batch_size
        epoch = resume.epoch if resume else 1
        skip = resume
        while True:
            yielded = False
            batch_start = skip.row_group_idx + step if skip is not None else start
            for batch_idx in range(batch_start, batch_count, step):
                offset = batch_idx * batch_size
                batch = docs[offset : offset + batch_size]
                position = PretrainCursor(
                    epoch=epoch,
                    shard_idx=0,
                    row_group_idx=batch_idx,
                )
                yielded = True
                yield batch, position
            was_skipping = skip is not None
            skip = None
            if not yielded and not was_skipping:
                msg = (
                    f"InMemoryTextSource yields no batches for split={split!r} "
                    f"with start={start}, step={step}"
                )
                raise ValueError(msg)
            epoch += 1
