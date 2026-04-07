"""Named pretraining datasets.

This is the only module that knows about Hugging Face; ``source.py`` stays
offline-pure so its tests never touch the network.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq

from nanodiffusion.data.source import ParquetTextSource, TextSource


@runtime_checkable
class DatasetFactory(Protocol):
    """Callable that builds a :class:`TextSource` from a local cache dir.

    ``download_options`` is honored by HTTP-backed factories and silently
    ignored by local-only ones so the CLI can stay generic.

    Named ``DatasetFactory`` rather than ``Dataset`` to avoid collision with
    ``torch.utils.data.Dataset`` and HuggingFace ``datasets.Dataset``.
    """

    def __call__(
        self,
        data_dir: Path,
        *,
        num_train: int | None = None,
        download: bool = True,
        download_options: "DownloadOptions | None" = None,
    ) -> TextSource: ...


DATASETS: dict[str, DatasetFactory] = {}


def register(name: str) -> Callable[[DatasetFactory], DatasetFactory]:
    def decorator(fn: DatasetFactory) -> DatasetFactory:
        if name in DATASETS:
            msg = f"Dataset {name!r} already registered"
            raise ValueError(msg)
        DATASETS[name] = fn
        return fn

    return decorator


def get_dataset(name: str) -> DatasetFactory:
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS)) or "(none)"
        msg = f"Unknown dataset {name!r}. Available: {available}"
        raise KeyError(msg)
    return DATASETS[name]


@dataclass(frozen=True, slots=True)
class DownloadOptions:
    retries: int = 5
    timeout: float = 60.0
    backoff_base: float = 2.0
    backoff_cap: float = 16.0
    backoff_jitter: float = 0.5
    chunk_size: int = 1 << 20
    num_workers: int = 4


def parquet_from_huggingface(
    *,
    repo_id: str,
    filename_pattern: str,
    train_indices: Iterable[int],
    val_indices: Iterable[int],
    data_dir: Path,
    download: bool = True,
    text_column: str = "text",
    options: DownloadOptions | None = None,
) -> ParquetTextSource:
    """Download any missing shards, then return a :class:`ParquetTextSource`.

    ``filename_pattern`` is a Python format string with one ``{index}``
    placeholder, e.g. ``"shard_{index:05d}.parquet"``.
    """
    opts = options or DownloadOptions()
    train_idx = list(train_indices)
    val_idx = list(val_indices)
    if download:
        _download_shards(
            repo_id=repo_id,
            filename_pattern=filename_pattern,
            indices=train_idx + val_idx,
            dest_dir=data_dir,
            options=opts,
        )
    train_paths = [data_dir / filename_pattern.format(index=i) for i in train_idx]
    val_paths = [data_dir / filename_pattern.format(index=i) for i in val_idx]
    return ParquetTextSource(train_paths, val_paths, text_column=text_column)


_HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def _download_shards(
    *,
    repo_id: str,
    filename_pattern: str,
    indices: list[int],
    dest_dir: Path,
    options: DownloadOptions,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        i for i in indices if not (dest_dir / filename_pattern.format(index=i)).exists()
    ]
    if not todo:
        return

    def task(index: int) -> None:
        filename = filename_pattern.format(index=index)
        url = _HF_RESOLVE_URL.format(repo_id=repo_id, filename=filename)
        target = dest_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        _download_with_backoff(
            url, target, options=options, validator=_validate_parquet
        )

    with ThreadPoolExecutor(max_workers=options.num_workers) as pool:
        for _ in pool.map(task, todo):
            pass


def _validate_parquet(path: Path) -> None:
    """Verify the file at ``path`` is a readable parquet file.

    Raises :class:`ValueError` so the download retry loop treats a corrupt
    payload (e.g. an HTML error page served with HTTP 200) the same as a
    transient network failure.
    """
    try:
        # Touch metadata so parquet reads the footer instead of just opening
        # the file handle.
        _ = pq.ParquetFile(path).metadata
    except (pa.ArrowException, OSError) as exc:
        msg = f"File at {path} is not a valid parquet"
        raise ValueError(msg) from exc


def _download_with_backoff(
    url: str,
    target: Path,
    *,
    options: DownloadOptions,
    validator: Callable[[Path], None] | None = None,
) -> None:
    # Local imports keep module import cheap for callers that only touch
    # the registry (e.g. ``data list``).
    import random  # noqa: PLC0415
    import time  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    import requests  # noqa: PLC0415

    # Per-call uuid lets concurrent downloads of the same shard write to
    # different temp files; the final rename is atomic.
    tmp = target.with_suffix(f"{target.suffix}.{uuid.uuid4().hex}.tmp")
    last_exc: Exception | None = None
    for attempt in range(1, options.retries + 1):
        try:
            with requests.get(url, stream=True, timeout=options.timeout) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=options.chunk_size):
                        if chunk:
                            fh.write(chunk)
            # Validate before rename so a corrupt payload is retried instead
            # of being committed and then failing at read time.
            if validator is not None:
                validator(tmp)
            tmp.replace(target)
        except (requests.RequestException, OSError, ValueError) as exc:
            last_exc = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < options.retries:
                # Capped exponential backoff with jitter to scatter a
                # thundering herd of failed shards.
                base = min(options.backoff_base**attempt, options.backoff_cap)
                jitter = random.uniform(0.0, options.backoff_jitter * base)  # noqa: S311
                time.sleep(base + jitter)
            continue
        else:
            return
    msg = f"Failed to download {url} after {options.retries} attempts"
    raise RuntimeError(msg) from last_exc


@dataclass(frozen=True, slots=True)
class _HFDataset:
    repo_id: str
    filename_pattern: str
    num_shards: int  # train shards; val shard(s) follow
    doc: str


_HF_REGISTRY: dict[str, _HFDataset] = {
    "climbmix-400b": _HFDataset(
        repo_id="karpathy/climbmix-400b-shuffle",
        filename_pattern="shard_{index:05d}.parquet",
        num_shards=6542,
        doc="ClimbMix-400B (Karpathy). nanochat default. 6543 shards, last is val.",
    ),
    "fineweb-edu-10bt": _HFDataset(
        repo_id="HuggingFaceFW/fineweb-edu",
        filename_pattern="sample/10BT/{index:03d}_00000.parquet",
        num_shards=13,
        doc="FineWeb-Edu sample-10BT subset (HuggingFaceFW). 14 shards, last is val.",
    ),
}


def _make_hf_factory(spec: _HFDataset) -> DatasetFactory:
    def factory(
        data_dir: Path,
        *,
        num_train: int | None = None,
        download: bool = True,
        download_options: DownloadOptions | None = None,
    ) -> ParquetTextSource:
        return parquet_from_huggingface(
            repo_id=spec.repo_id,
            filename_pattern=spec.filename_pattern,
            train_indices=range(
                num_train if num_train is not None else spec.num_shards
            ),
            val_indices=(spec.num_shards,),
            data_dir=data_dir,
            download=download,
            options=download_options,
        )

    factory.__doc__ = spec.doc
    return factory


for _name, _spec in _HF_REGISTRY.items():
    register(_name)(_make_hf_factory(_spec))
