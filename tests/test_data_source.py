from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanodiffusion.data.source import (
    InMemoryTextSource,
    ParquetTextSource,
    SourcePosition,
    TextSource,
)


def _take[T](it: Iterator[T], n: int) -> list[T]:
    out: list[T] = []
    for i, item in enumerate(it):
        if i >= n:
            break
        out.append(item)
    return out


def test_in_memory_source_satisfies_protocol() -> None:
    src = InMemoryTextSource(["a", "b", "c"], val_size=1)
    assert isinstance(src, TextSource)


def test_in_memory_source_train_val_split() -> None:
    src = InMemoryTextSource(["a", "b", "c", "d"], val_size=1)

    train_first = next(src.iter_documents("train", batch_size=10))
    val_first = next(src.iter_documents("val", batch_size=10))

    assert train_first[0] == ["a", "b", "c"]
    assert val_first[0] == ["d"]


def test_in_memory_source_batches_respect_batch_size() -> None:
    src = InMemoryTextSource([f"doc{i}" for i in range(5)], val_size=1)
    batches = _take(src.iter_documents("train", batch_size=2), 3)

    assert [b[0] for b in batches] == [
        ["doc0", "doc1"],
        ["doc2", "doc3"],
        ["doc0", "doc1"],
    ]
    # epoch advances on the third batch (we wrapped around)
    assert batches[0][1]["epoch"] == 1
    assert batches[1][1]["epoch"] == 1
    assert batches[2][1]["epoch"] == 2


def test_in_memory_source_position_monotonic_within_epoch() -> None:
    src = InMemoryTextSource(["a", "b", "c", "d", "e"], val_size=1)
    batches = _take(src.iter_documents("train", batch_size=1), 4)

    epochs = [b[1]["epoch"] for b in batches]
    rg_idx = [b[1]["row_group_idx"] for b in batches]
    assert epochs == [1, 1, 1, 1]
    assert rg_idx == sorted(rg_idx)


def test_in_memory_source_rejects_invalid_val_size() -> None:
    with pytest.raises(ValueError, match="val_size"):
        InMemoryTextSource(["a", "b"], val_size=2)
    with pytest.raises(ValueError, match="val_size"):
        InMemoryTextSource(["a", "b"], val_size=0)


def _write_parquet(path: Path, texts: list[str], row_group_size: int = 2) -> None:
    table = pa.table({"text": texts})
    pq.write_table(table, path, row_group_size=row_group_size)


def test_parquet_source_round_trip(tmp_path: Path) -> None:
    train_a = tmp_path / "train_a.parquet"
    train_b = tmp_path / "train_b.parquet"
    val = tmp_path / "val.parquet"
    _write_parquet(train_a, ["doc0", "doc1", "doc2", "doc3"], row_group_size=2)
    _write_parquet(train_b, ["doc4", "doc5"], row_group_size=2)
    _write_parquet(val, ["v0", "v1"], row_group_size=1)

    src = ParquetTextSource([train_a, train_b], [val])

    train_batches = _take(src.iter_documents("train", batch_size=10), 3)
    flat_train = [doc for b in train_batches for doc in b[0]]
    assert flat_train[:6] == ["doc0", "doc1", "doc2", "doc3", "doc4", "doc5"]

    val_batches = _take(src.iter_documents("val", batch_size=10), 1)
    assert val_batches[0][0] == ["v0"]


def test_parquet_source_position_advances(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    _write_parquet(train, ["a", "b", "c", "d"], row_group_size=2)
    _write_parquet(val, ["v"], row_group_size=1)

    src = ParquetTextSource([train], [val])
    positions: list[SourcePosition] = [
        b[1] for b in _take(src.iter_documents("train", batch_size=2), 2)
    ]
    assert positions[0]["shard_idx"] == 0
    assert positions[1]["shard_idx"] == 0
    assert positions[0]["row_group_idx"] == 0
    assert positions[1]["row_group_idx"] == 1


def test_parquet_source_loops_and_increments_epoch(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    _write_parquet(train, ["a", "b"], row_group_size=2)
    _write_parquet(val, ["v"], row_group_size=1)

    src = ParquetTextSource([train], [val])
    batches = _take(src.iter_documents("train", batch_size=10), 3)
    epochs = [b[1]["epoch"] for b in batches]
    assert epochs[0] == 1
    assert epochs[-1] == 3


def test_parquet_source_rejects_empty_split(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    _write_parquet(train, ["a"], row_group_size=1)
    src = ParquetTextSource([train], [])

    with pytest.raises(ValueError, match="val"):
        next(src.iter_documents("val"))
