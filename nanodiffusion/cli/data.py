"""``nanodiffusion data ...`` commands: list and download pretraining datasets."""

from pathlib import Path

import click


@click.group(name="data")
def data_group() -> None:
    """Data pipeline commands."""


@data_group.command(name="list")
def list_datasets() -> None:
    """List registered datasets."""
    from nanodiffusion.data.datasets import DATASETS  # noqa: PLC0415

    for name in sorted(DATASETS):
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
def download(dataset: str, num_train: int, data_dir: Path) -> None:
    """Download parquet shards for a registered dataset."""
    from nanodiffusion.data.datasets import get  # noqa: PLC0415

    factory = get(dataset)
    factory(data_dir, num_train=num_train, download=True)
    click.echo(f"Downloaded {num_train} train shards + val shard(s) to {data_dir}")
