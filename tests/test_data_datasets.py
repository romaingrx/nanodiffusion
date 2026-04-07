from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanodiffusion.data.datasets import (
    DATASETS,
    Dataset,
    climbmix_400b,
    fineweb_edu_10bt,
    get,
    parquet_from_huggingface,
    register,
)
from nanodiffusion.data.source import ParquetTextSource


def test_default_registry_entries() -> None:
    assert "climbmix-400b" in DATASETS
    assert "fineweb-edu-10bt" in DATASETS


def test_registered_factories_satisfy_protocol() -> None:
    assert isinstance(climbmix_400b, Dataset)
    assert isinstance(fineweb_edu_10bt, Dataset)


def test_register_decorator_adds_entry() -> None:
    name = "test-register-add"
    try:

        @register(name)
        def my_dataset(
            data_dir: Path,
            *,
            num_train: int | None = None,
            download: bool = True,
        ) -> ParquetTextSource:
            del data_dir, num_train, download
            raise NotImplementedError

        assert name in DATASETS
        assert get(name) is my_dataset
    finally:
        DATASETS.pop(name, None)


def test_register_rejects_duplicate() -> None:
    name = "test-register-dup"
    try:

        @register(name)
        def first(
            data_dir: Path,
            *,
            num_train: int | None = None,
            download: bool = True,
        ) -> ParquetTextSource:
            del data_dir, num_train, download
            raise NotImplementedError

        with pytest.raises(ValueError, match="already registered"):

            @register(name)
            def second(
                data_dir: Path,
                *,
                num_train: int | None = None,
                download: bool = True,
            ) -> ParquetTextSource:
                del data_dir, num_train, download
                raise NotImplementedError

    finally:
        DATASETS.pop(name, None)


def test_get_unknown_lists_available() -> None:
    with pytest.raises(KeyError, match="climbmix-400b"):
        get("definitely-does-not-exist")


def _write_parquet(path: Path, texts: list[str]) -> None:
    table = pa.table({"text": texts})
    pq.write_table(table, path, row_group_size=2)


def test_parquet_from_huggingface_no_download(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "shard_00000.parquet", ["doc0", "doc1"])
    _write_parquet(tmp_path / "shard_00001.parquet", ["doc2", "doc3"])
    _write_parquet(tmp_path / "shard_00099.parquet", ["v0"])

    src = parquet_from_huggingface(
        repo_id="ignored",
        filename_pattern="shard_{index:05d}.parquet",
        train_indices=[0, 1],
        val_indices=[99],
        data_dir=tmp_path,
        download=False,
    )
    assert isinstance(src, ParquetTextSource)
    train_first = next(src.iter_documents("train", batch_size=10))
    assert "doc0" in train_first[0]


def test_parquet_from_huggingface_downloads_with_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First two HTTP attempts fail, third succeeds."""
    fixture = tmp_path / "fixture.parquet"
    _write_parquet(fixture, ["alpha", "beta"])

    call_count = {"n": 0}

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int) -> Iterator[bytes]:
            del chunk_size
            yield self._payload

    def fake_get(_url: str, **_kwargs: Any) -> FakeResponse:
        call_count["n"] += 1
        if call_count["n"] < 3:
            import requests as _requests  # noqa: PLC0415

            raise _requests.RequestException("simulated transient failure")
        return FakeResponse(fixture.read_bytes())

    import requests  # noqa: PLC0415

    monkeypatch.setattr(requests, "get", fake_get)
    # The lazy `import time` inside _download_with_backoff resolves to the
    # global time module, so patching time.sleep here suppresses the backoff.
    import time  # noqa: PLC0415

    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    dest = tmp_path / "out"
    src = parquet_from_huggingface(
        repo_id="user/repo",
        filename_pattern="shard_{index:05d}.parquet",
        train_indices=[0],
        val_indices=[1],
        data_dir=dest,
        download=True,
        num_workers=1,
    )

    assert call_count["n"] >= 3
    assert (dest / "shard_00000.parquet").exists()
    assert (dest / "shard_00001.parquet").exists()
    train_first = next(src.iter_documents("train", batch_size=10))
    assert train_first[0] == ["alpha", "beta"]


def test_parquet_from_huggingface_skips_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_parquet(tmp_path / "shard_00000.parquet", ["existing"])
    _write_parquet(tmp_path / "shard_00099.parquet", ["existing-val"])

    def explode(*_a: object, **_k: object) -> None:
        msg = "should not be called"
        raise AssertionError(msg)

    import requests  # noqa: PLC0415

    monkeypatch.setattr(requests, "get", explode)

    src = parquet_from_huggingface(
        repo_id="user/repo",
        filename_pattern="shard_{index:05d}.parquet",
        train_indices=[0],
        val_indices=[99],
        data_dir=tmp_path,
        download=True,
    )
    train_first = next(src.iter_documents("train", batch_size=10))
    assert train_first[0] == ["existing"]
