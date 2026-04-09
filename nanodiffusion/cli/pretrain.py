"""``nanodiffusion pretrain`` command: run an MDLM pretraining job."""

from pathlib import Path

import click


@click.command(name="pretrain")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a YAML config matching nanodiffusion.config.Config.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Override train.seed from the config (useful for multi-seed sweeps).",
)
@click.option(
    "--resume-from",
    "resume_from",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Checkpoint directory to resume from. Can be an explicit step "
        "dir (runs/pretrain/<id>/step_1000) or the 'latest' symlink "
        "(runs/pretrain/<id>/latest)."
    ),
)
def pretrain_command(
    *,
    config_path: Path,
    seed: int | None,
    resume_from: Path | None,
) -> None:
    """Run MDLM pretraining end-to-end."""
    from nanodiffusion.config import Config  # noqa: PLC0415
    from nanodiffusion.pretrain import pretrain  # noqa: PLC0415

    config = Config.from_yaml(config_path)
    if seed is not None:
        config.train.seed = seed
    pretrain(config, resume_from=resume_from)
