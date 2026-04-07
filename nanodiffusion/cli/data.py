"""``nanodiffusion data ...`` commands: list and download pretraining datasets."""

from pathlib import Path

import click


@click.group(name="data")
def data_group() -> None:
    """Data pipeline commands."""


@data_group.command(name="list")
def list_datasets() -> None:
    """List registered datasets with one-line descriptions."""
    from nanodiffusion.data.datasets import DATASETS  # noqa: PLC0415

    for name in sorted(DATASETS):
        factory = DATASETS[name]
        first_line = (factory.__doc__ or "").strip().split("\n", maxsplit=1)[0]
        if first_line:
            click.echo(f"{name}\t{first_line}")
        else:
            click.echo(name)


@data_group.command()
@click.option(
    "--dataset",
    default="climbmix-400b",
    show_default=True,
    help="Registered dataset name (see `nanodiffusion data list`)",
)
@click.option(
    "--num-train",
    type=int,
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
    from nanodiffusion.data.datasets import (  # noqa: PLC0415
        DATASETS,
        DownloadOptions,
        get_dataset,
    )

    if dataset not in DATASETS:
        available = ", ".join(sorted(DATASETS)) or "(none)"
        msg = f"Unknown dataset {dataset!r}. Available: {available}"
        raise click.BadParameter(msg, param_hint="--dataset")
    factory = get_dataset(dataset)
    options = DownloadOptions(retries=retries, timeout=timeout, num_workers=num_workers)
    factory(
        data_dir,
        num_train=num_train,
        download=True,
        download_options=options,
    )
    click.echo(f"Downloaded {num_train} train shards + val shard(s) to {data_dir}")
