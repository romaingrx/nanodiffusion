"""``nanodiffusion sft`` command: fine-tune a pretrained checkpoint."""

from pathlib import Path

import click


@click.command(name="sft")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a YAML config matching nanodiffusion.config.Config.",
)
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help=(
        "Directory of a pretrain checkpoint to fine-tune from. Can be an "
        "explicit step dir (runs/pretrain/<id>/step_1000) or the 'latest' "
        "symlink (runs/pretrain/<id>/latest)."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Override sft.seed from the config (useful for multi-seed sweeps).",
)
def sft_command(
    *,
    config_path: Path,
    checkpoint: Path,
    seed: int | None,
) -> None:
    """Run SFT on a pretrained checkpoint end-to-end."""
    from nanodiffusion.config import Config  # noqa: PLC0415
    from nanodiffusion.sft import sft_finetune  # noqa: PLC0415

    config = Config.from_yaml(config_path)
    if seed is not None:
        config.sft.seed = seed
    sft_finetune(config, checkpoint=checkpoint)
