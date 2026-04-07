"""Named pretraining datasets.

A ``Dataset`` is a callable that materializes a :class:`TextSource` on
demand. Datasets are registered in the module-level :data:`DATASETS` dict
via the :func:`register` decorator. The config and CLI reference datasets
by name; adding a new corpus is one decorated function below.

This is the only module that knows about Hugging Face. ``source.py`` stays
offline-pure so its tests never touch the network.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, runtime_checkable

from nanodiffusion.data.source import ParquetTextSource, TextSource


@runtime_checkable
class Dataset(Protocol):
    """Callable that builds a :class:`TextSource` from a local cache dir.

    Implementations may download files, scan a local directory, or build a
    synthetic source. ``num_train`` lets the caller request a smaller slice
    than the dataset's full train set; ``None`` means use whatever the
    dataset considers full.
    """

    def __call__(
        self,
        data_dir: Path,
        *,
        num_train: int | None = None,
        download: bool = True,
    ) -> TextSource: ...


DATASETS: dict[str, Dataset] = {}


def register(name: str) -> Callable[[Dataset], Dataset]:
    """Decorator: register a dataset factory under ``name``.

    Raises :class:`ValueError` on duplicate names so silent shadowing
    cannot happen if two modules define the same key.
    """

    def decorator(fn: Dataset) -> Dataset:
        if name in DATASETS:
            msg = f"Dataset {name!r} already registered"
            raise ValueError(msg)
        DATASETS[name] = fn
        return fn

    return decorator


def get(name: str) -> Dataset:
    """Look up a registered dataset by name."""
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS)) or "(none)"
        msg = f"Unknown dataset {name!r}. Available: {available}"
        raise KeyError(msg)
    return DATASETS[name]


def parquet_from_huggingface(
    *,
    repo_id: str,
    filename_pattern: str,
    train_indices: Iterable[int],
    val_indices: Iterable[int],
    data_dir: Path,
    download: bool = True,
    text_column: str = "text",
    num_workers: int = 4,
) -> ParquetTextSource:
    """Helper most HF parquet datasets reuse.

    Downloads any missing shards (if ``download``) into ``data_dir``, then
    returns a :class:`ParquetTextSource` pointing at the resulting paths.
    ``filename_pattern`` is a Python format string with one ``{index}``
    placeholder, e.g. ``"shard_{index:05d}.parquet"``.
    """
    train_idx = list(train_indices)
    val_idx = list(val_indices)
    if download:
        _download_shards(
            repo_id=repo_id,
            filename_pattern=filename_pattern,
            indices=train_idx + val_idx,
            dest_dir=data_dir,
            num_workers=num_workers,
        )
    train_paths = [data_dir / filename_pattern.format(index=i) for i in train_idx]
    val_paths = [data_dir / filename_pattern.format(index=i) for i in val_idx]
    return ParquetTextSource(train_paths, val_paths, text_column=text_column)


_HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
_DOWNLOAD_RETRIES = 5


def _download_shards(
    *,
    repo_id: str,
    filename_pattern: str,
    indices: list[int],
    dest_dir: Path,
    num_workers: int,
) -> None:
    """Download missing parquet shards from a HF dataset repo.

    Atomic via temp-file rename. Skips files already on disk. Parallelized
    with a ``ThreadPoolExecutor``. The ``requests`` import is local so
    importing this module stays cheap for callers that only touch the
    registry (e.g. ``data list``).
    """
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
        _download_with_backoff(url, target)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for _ in pool.map(task, todo):
            pass


def _download_with_backoff(url: str, target: Path) -> None:
    import time  # noqa: PLC0415

    import requests  # noqa: PLC0415

    tmp = target.with_suffix(target.suffix + ".tmp")
    last_exc: Exception | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(target)
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < _DOWNLOAD_RETRIES:
                time.sleep(2**attempt)
            continue
        else:
            return
    msg = f"Failed to download {url} after {_DOWNLOAD_RETRIES} attempts"
    raise RuntimeError(msg) from last_exc


@register("climbmix-400b")
def climbmix_400b(
    data_dir: Path,
    *,
    num_train: int | None = None,
    download: bool = True,
) -> ParquetTextSource:
    """ClimbMix-400B (Karpathy). nanochat default. 6543 shards, last is val."""
    return parquet_from_huggingface(
        repo_id="karpathy/climbmix-400b-shuffle",
        filename_pattern="shard_{index:05d}.parquet",
        train_indices=range(num_train if num_train is not None else 6542),
        val_indices=(6542,),
        data_dir=data_dir,
        download=download,
    )


@register("fineweb-edu-10bt")
def fineweb_edu_10bt(
    data_dir: Path,
    *,
    num_train: int | None = None,
    download: bool = True,
) -> ParquetTextSource:
    """FineWeb-Edu sample-10BT subset (HuggingFaceFW).

    The exact filename pattern needs to be confirmed against the live repo
    layout the first time this dataset is used; the placeholder below
    follows the conventional ``sample/10BT/000_<NN>.parquet`` shape.
    """
    return parquet_from_huggingface(
        repo_id="HuggingFaceFW/fineweb-edu",
        filename_pattern="sample/10BT/000_{index:05d}.parquet",
        train_indices=range(num_train if num_train is not None else 14),
        val_indices=(14,),
        data_dir=data_dir,
        download=download,
    )
