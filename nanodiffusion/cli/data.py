"""``nanodiffusion data ...`` commands: list and download datasets."""

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path

import click


@click.group(name="data")
def data_group() -> None:
    """Data pipeline commands."""


def _echo_registry(registry: Mapping[str, Callable[..., object]]) -> None:
    for name in sorted(registry):
        doc = inspect.getdoc(registry[name]) or ""
        first_line = doc.partition("\n")[0]
        click.echo(f"{name}\t{first_line}" if first_line else name)


@data_group.command(name="list")
def list_datasets() -> None:
    """List registered pretraining datasets with one-line descriptions."""
    from nanodiffusion.data.datasets import DATASETS

    _echo_registry(DATASETS)


@data_group.command(name="list-chat")
def list_chat_datasets() -> None:
    """List registered chat (SFT) datasets with one-line descriptions."""
    from nanodiffusion.data.chat_datasets import CHAT_DATASETS

    _echo_registry(CHAT_DATASETS)


@data_group.command()
@click.option(
    "--dataset",
    default="climbmix-400b",
    show_default=True,
    help="Registered dataset name (see `nanodiffusion data list`)",
)
@click.option(
    "--num-train",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of train shards to download",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("data"),
    show_default=True,
)
@click.option(
    "--retries",
    type=int,
    default=5,
    show_default=True,
    help="Max retries per shard before failing",
)
@click.option(
    "--timeout",
    type=float,
    default=60.0,
    show_default=True,
    help="HTTP request timeout in seconds",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    show_default=True,
    help="Parallel download workers",
)
def download(
    dataset: str,
    num_train: int,
    data_dir: Path,
    retries: int,
    timeout: float,
    num_workers: int,
) -> None:
    """Download parquet shards for a registered dataset."""
    from nanodiffusion.data.datasets import (
        DownloadOptions,
        get_dataset,
    )

    try:
        factory = get_dataset(dataset)
    except KeyError as exc:
        raise click.BadParameter(exc.args[0], param_hint="--dataset") from exc
    options = DownloadOptions(retries=retries, timeout=timeout, num_workers=num_workers)
    factory(
        data_dir,
        num_train=num_train,
        download=True,
        download_options=options,
    )
    click.echo(f"Downloaded {num_train} train shards + val shard(s) to {data_dir}")


@data_group.command(name="download-chat")
@click.option(
    "--dataset",
    default="smoltalk",
    show_default=True,
    help="Registered chat dataset name (see `nanodiffusion data list-chat`).",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("data"),
    show_default=True,
)
@click.option(
    "--retries",
    type=int,
    default=5,
    show_default=True,
    help="Max retries per request for the identity JSONL fetch",
)
@click.option(
    "--timeout",
    type=float,
    default=60.0,
    show_default=True,
    help="HTTP request timeout in seconds for the identity JSONL fetch",
)
def download_chat(
    dataset: str,
    data_dir: Path,
    retries: int,
    timeout: float,
) -> None:
    """Download an SFT chat dataset into ``data-dir``.

    HuggingFace-hosted datasets (smoltalk, gsm8k) are cached via the
    ``datasets`` library; the identity conversations bundle goes
    through the small retry/backoff downloader in
    :mod:`nanodiffusion.data.datasets`. ``--retries`` and ``--timeout``
    therefore only affect the identity path.
    """
    from nanodiffusion.data.chat_datasets import get_chat_dataset
    from nanodiffusion.data.datasets import DownloadOptions

    try:
        factory = get_chat_dataset(dataset)
    except KeyError as exc:
        raise click.BadParameter(exc.args[0], param_hint="--dataset") from exc
    options = DownloadOptions(retries=retries, timeout=timeout)
    factory(data_dir, download=True, download_options=options)
    click.echo(f"Downloaded chat dataset {dataset!r} to {data_dir}")
