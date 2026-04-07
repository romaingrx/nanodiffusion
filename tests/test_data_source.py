from itertools import islice
from pathlib import Path

import pytest

from nanodiffusion.data.source import (
    InMemoryTextSource,
    ParquetTextSource,
    SourcePosition,
    TextSource,
)
from tests._helpers import write_parquet


def test_in_memory_source_satisfies_protocol_structurally() -> None:
    """Without @runtime_checkable we rely on duck typing; just call the method."""
    src = InMemoryTextSource(["a", "b", "c"], val_size=1)
    _: TextSource = src  # type: ignore[assignment]  # static-only check
    next(src.iter_documents("train", batch_size=1))


def test_in_memory_source_train_val_split() -> None:
    src = InMemoryTextSource(["a", "b", "c", "d"], val_size=1)

    train_first = next(src.iter_documents("train", batch_size=10))
    val_first = next(src.iter_documents("val", batch_size=10))

    assert train_first[0] == ["a", "b", "c"]
    assert val_first[0] == ["d"]


def test_in_memory_source_batches_respect_batch_size() -> None:
    src = InMemoryTextSource([f"doc{i}" for i in range(5)], val_size=1)
    batches = list(islice(src.iter_documents("train", batch_size=2), 3))

    assert [b[0] for b in batches] == [
        ["doc0", "doc1"],
        ["doc2", "doc3"],
        ["doc0", "doc1"],
    ]
    assert batches[0][1]["epoch"] == 1
    assert batches[1][1]["epoch"] == 1
    assert batches[2][1]["epoch"] == 2


def test_in_memory_source_row_group_idx_is_batch_index() -> None:
    """The position uses batch-index semantics, matching parquet."""
    src = InMemoryTextSource([f"doc{i}" for i in range(8)], val_size=1)
    batches = list(islice(src.iter_documents("train", batch_size=2), 4))
    rg_indices = [b[1]["row_group_idx"] for b in batches]
    # 7 train docs in batches of 2 -> batch_count = 4 (last batch has 1 doc)
    assert rg_indices == [0, 1, 2, 3]


def test_in_memory_source_start_step_partition_is_disjoint() -> None:
    """Same (start, step) pair across two iterations gives a disjoint cover."""
    docs = [f"doc{i}" for i in range(11)]
    src = InMemoryTextSource(docs, val_size=1)

    even_batches = list(
        islice(src.iter_documents("train", batch_size=1, start=0, step=2), 5)
    )
    odd_batches = list(
        islice(src.iter_documents("train", batch_size=1, start=1, step=2), 5)
    )

    even = {tuple(b[0]) for b in even_batches}
    odd = {tuple(b[0]) for b in odd_batches}
    assert even.isdisjoint(odd)
    assert even | odd == {(d,) for d in docs[:-1]}  # all train docs covered


def test_in_memory_source_rejects_invalid_val_size() -> None:
    with pytest.raises(ValueError, match="val_size"):
        InMemoryTextSource(["a", "b"], val_size=2)
    with pytest.raises(ValueError, match="val_size"):
        InMemoryTextSource(["a", "b"], val_size=0)


def test_parquet_source_round_trip(tmp_path: Path) -> None:
    train_a = tmp_path / "train_a.parquet"
    train_b = tmp_path / "train_b.parquet"
    val = tmp_path / "val.parquet"
    write_parquet(train_a, ["doc0", "doc1", "doc2", "doc3"], row_group_size=2)
    write_parquet(train_b, ["doc4", "doc5"], row_group_size=2)
    write_parquet(val, ["v0", "v1"], row_group_size=1)

    src = ParquetTextSource([train_a, train_b], [val])

    train_batches = list(islice(src.iter_documents("train", batch_size=10), 3))
    flat_train = [doc for b in train_batches for doc in b[0]]
    assert flat_train[:6] == ["doc0", "doc1", "doc2", "doc3", "doc4", "doc5"]

    val_batches = list(islice(src.iter_documents("val", batch_size=10), 1))
    assert val_batches[0][0] == ["v0"]


def test_parquet_source_filters_null_rows(tmp_path: Path) -> None:
    """Null entries from parquet must not surface as None in the batch."""
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    write_parquet(train, ["a", None, "b", None, "c"], row_group_size=10)
    write_parquet(val, ["v"], row_group_size=1)

    src = ParquetTextSource([train], [val])
    batch = next(src.iter_documents("train", batch_size=10))
    assert batch[0] == ["a", "b", "c"]


def test_parquet_source_position_advances(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    write_parquet(train, ["a", "b", "c", "d"], row_group_size=2)
    write_parquet(val, ["v"], row_group_size=1)

    src = ParquetTextSource([train], [val])
    positions: list[SourcePosition] = [
        b[1] for b in islice(src.iter_documents("train", batch_size=2), 2)
    ]
    assert positions[0]["shard_idx"] == 0
    assert positions[1]["shard_idx"] == 0
    assert positions[0]["row_group_idx"] == 0
    assert positions[1]["row_group_idx"] == 1


def test_parquet_source_loops_and_increments_epoch(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    write_parquet(train, ["a", "b"], row_group_size=2)
    write_parquet(val, ["v"], row_group_size=1)

    src = ParquetTextSource([train], [val])
    batches = list(islice(src.iter_documents("train", batch_size=10), 3))
    epochs = [b[1]["epoch"] for b in batches]
    assert epochs[0] == 1
    assert epochs[-1] == 3


def test_parquet_source_start_step_partition_is_disjoint(tmp_path: Path) -> None:
    """Row-group-level striding produces non-overlapping partitions."""
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    # 6 docs, 1 doc per row group => 6 row groups in this shard.
    write_parquet(train, ["d0", "d1", "d2", "d3", "d4", "d5"], row_group_size=1)
    write_parquet(val, ["v"], row_group_size=1)

    src = ParquetTextSource([train], [val])
    even_batches = list(
        islice(src.iter_documents("train", batch_size=10, start=0, step=2), 3)
    )
    odd_batches = list(
        islice(src.iter_documents("train", batch_size=10, start=1, step=2), 3)
    )

    even_docs = {d for b in even_batches for d in b[0]}
    odd_docs = {d for b in odd_batches for d in b[0]}
    assert even_docs.isdisjoint(odd_docs)
    assert even_docs | odd_docs == {f"d{i}" for i in range(6)}


def test_parquet_source_custom_text_column(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    write_parquet(train, ["a", "b"], row_group_size=2, column="content")
    write_parquet(val, ["v"], row_group_size=1, column="content")

    src = ParquetTextSource([train], [val], text_column="content")
    batch = next(src.iter_documents("train", batch_size=10))
    assert batch[0] == ["a", "b"]


def test_parquet_source_rejects_empty_split(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    write_parquet(train, ["a"], row_group_size=1)
    src = ParquetTextSource([train], [])

    with pytest.raises(ValueError, match="val"):
        next(src.iter_documents("val"))
