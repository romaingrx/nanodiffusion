from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import pytest

from nanodiffusion.data.datasets import (
    DATASETS,
    DatasetFactory,
    DownloadOptions,
    get_dataset,
    parquet_from_huggingface,
    register,
)
from nanodiffusion.data.source import ParquetTextSource
from tests._helpers import write_parquet


def _dummy_factory(
    data_dir: Path,
    *,
    num_train: int | None = None,
    download: bool = True,
) -> ParquetTextSource:
    del data_dir, num_train, download
    raise NotImplementedError


def test_default_registry_entries() -> None:
    assert "climbmix-400b" in DATASETS
    assert "fineweb-edu-10bt" in DATASETS


def test_registered_factories_satisfy_protocol() -> None:
    assert isinstance(DATASETS["climbmix-400b"], DatasetFactory)
    assert isinstance(DATASETS["fineweb-edu-10bt"], DatasetFactory)


def test_register_decorator_adds_entry() -> None:
    name = "test-register-add"
    try:
        wrapped = register(name)(_dummy_factory)
        assert name in DATASETS
        assert get_dataset(name) is wrapped
    finally:
        DATASETS.pop(name, None)


def test_register_returns_original_function() -> None:
    name = "test-register-returns"
    try:
        wrapped = register(name)(_dummy_factory)
        assert wrapped is _dummy_factory
    finally:
        DATASETS.pop(name, None)


def test_register_rejects_duplicate() -> None:
    name = "test-register-dup"
    try:
        register(name)(_dummy_factory)
        with pytest.raises(ValueError, match="already registered"):
            register(name)(_dummy_factory)
    finally:
        DATASETS.pop(name, None)


def test_get_dataset_unknown_lists_available() -> None:
    """Error message must contain at least one currently-registered name."""
    with pytest.raises(KeyError) as exc_info:
        get_dataset("definitely-does-not-exist")
    msg = str(exc_info.value)
    assert "definitely-does-not-exist" in msg
    # Whatever default datasets are registered, at least one should be listed.
    assert any(name in msg for name in DATASETS)


def test_get_dataset_empty_registry_message() -> None:
    """When DATASETS is empty the error must say so cleanly."""
    saved = dict(DATASETS)
    try:
        DATASETS.clear()
        with pytest.raises(KeyError, match=r"\(none\)"):
            get_dataset("anything")
    finally:
        DATASETS.update(saved)


def test_parquet_from_huggingface_no_download(tmp_path: Path) -> None:
    write_parquet(tmp_path / "shard_00000.parquet", ["doc0", "doc1"])
    write_parquet(tmp_path / "shard_00001.parquet", ["doc2", "doc3"])
    write_parquet(tmp_path / "shard_00099.parquet", ["v0"])

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


def _no_sleep_options(retries: int = 5) -> DownloadOptions:
    """DownloadOptions tuned to skip wall-clock waits in tests."""
    return DownloadOptions(
        retries=retries,
        timeout=5.0,
        backoff_base=0.0,
        backoff_cap=0.0,
        backoff_jitter=0.0,
    )


class _FakeResponse:
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


def test_download_succeeds_on_first_try(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture.parquet"
    write_parquet(fixture, ["alpha", "beta"])

    calls = {"n": 0}

    def fake_get(_url: str, **_kwargs: Any) -> _FakeResponse:
        calls["n"] += 1
        return _FakeResponse(fixture.read_bytes())

    import requests  # noqa: PLC0415

    monkeypatch.setattr(requests, "get", fake_get)

    dest = tmp_path / "out"
    parquet_from_huggingface(
        repo_id="user/repo",
        filename_pattern="shard_{index:05d}.parquet",
        train_indices=[0],
        val_indices=[1],
        data_dir=dest,
        download=True,
        options=_no_sleep_options(),
    )
    assert calls["n"] == 2
    assert (dest / "shard_00000.parquet").exists()


def test_download_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First two HTTP attempts fail, third succeeds."""
    fixture = tmp_path / "fixture.parquet"
    write_parquet(fixture, ["alpha", "beta"])

    calls = {"n": 0}

    def fake_get(_url: str, **_kwargs: Any) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] < 3:
            import requests as _requests  # noqa: PLC0415

            raise _requests.RequestException("simulated transient failure")
        return _FakeResponse(fixture.read_bytes())

    import requests  # noqa: PLC0415

    monkeypatch.setattr(requests, "get", fake_get)

    dest = tmp_path / "out"
    src = parquet_from_huggingface(
        repo_id="user/repo",
        filename_pattern="shard_{index:05d}.parquet",
        train_indices=[0],
        val_indices=[],
        data_dir=dest,
        download=True,
        options=_no_sleep_options(),
    )
    assert calls["n"] == 3
    assert (dest / "shard_00000.parquet").exists()
    train_first = next(src.iter_documents("train", batch_size=10))
    assert train_first[0] == ["alpha", "beta"]


def test_download_total_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_fail(_url: str, **_kwargs: Any) -> None:
        import requests as _requests  # noqa: PLC0415

        raise _requests.RequestException("permanent failure")

    import requests  # noqa: PLC0415

    monkeypatch.setattr(requests, "get", always_fail)

    with pytest.raises(RuntimeError, match="Failed to download"):
        parquet_from_huggingface(
            repo_id="user/repo",
            filename_pattern="shard_{index:05d}.parquet",
            train_indices=[0],
            val_indices=[],
            data_dir=tmp_path / "out",
            download=True,
            options=_no_sleep_options(retries=2),
        )

    # No final shard file should be left on disk after total failure.
    assert not (tmp_path / "out" / "shard_00000.parquet").exists()
    # And no temp files should be left behind.
    leftover = list((tmp_path / "out").glob("*.tmp"))
    assert leftover == []


def test_download_skips_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_parquet(tmp_path / "shard_00000.parquet", ["existing"])
    write_parquet(tmp_path / "shard_00099.parquet", ["existing-val"])

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
        options=_no_sleep_options(),
    )
    train_first = next(src.iter_documents("train", batch_size=10))
    assert train_first[0] == ["existing"]
