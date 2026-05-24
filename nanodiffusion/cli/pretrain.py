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
        "Run directory to resume from (e.g. runs/pretrain/<id>). The "
        "latest finalised step is selected automatically by the Orbax "
        "checkpoint manager — no step subdir is needed."
    ),
)
@click.option(
    "--wandb-project",
    envvar="WANDB_PROJECT",
    default=None,
    help=(
        "Enable wandb logging under the given project. Also reads "
        "WANDB_PROJECT from the environment. Requires `pip install "
        "nanodiffusion[obs]` for the wandb dependency."
    ),
)
@click.option(
    "--wandb-entity",
    envvar="WANDB_ENTITY",
    default=None,
    help="Optional wandb entity (team/user) scope for the run.",
)
@click.option(
    "--profile-steps",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Profile this many steps after the first JIT compile and save "
        "the trace to run_dir/profile/ (viewable in TensorBoard)."
    ),
)
def pretrain_command(
    *,
    config_path: Path,
    seed: int | None,
    resume_from: Path | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    profile_steps: int,
) -> None:
    """Run MDLM pretraining end-to-end."""
    from nanodiffusion.config import Config
    from nanodiffusion.pretrain import pretrain

    config = Config.from_yaml(config_path)
    if seed is not None:
        config.train.seed = seed
    pretrain(
        config,
        resume_from=resume_from,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        profile_steps=profile_steps,
    )
