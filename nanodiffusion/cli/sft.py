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
    "--pretrain-checkpoint",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory of a pretrain checkpoint to start fine-tuning from. "
        "Mutually exclusive with --resume-from. Can be an explicit step "
        "dir (runs/pretrain/<id>/step_1000) or the 'latest' symlink."
    ),
)
@click.option(
    "--resume-from",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory of an interrupted SFT checkpoint to resume. "
        "Mutually exclusive with --pretrain-checkpoint. The run "
        "directory, optimizer state, EMA, step counter, and loader "
        "cursor are all carried over from the saved run."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Override sft.seed from the config (useful for multi-seed sweeps).",
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
def sft_command(
    *,
    config_path: Path,
    pretrain_checkpoint: Path | None,
    resume_from: Path | None,
    seed: int | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    profile_steps: int,
) -> None:
    """Run SFT from a pretrain checkpoint or resume an interrupted run."""
    from nanodiffusion.config import Config
    from nanodiffusion.sft import sft_finetune

    if (pretrain_checkpoint is None) == (resume_from is None):
        msg = "pass exactly one of --pretrain-checkpoint or --resume-from"
        raise click.UsageError(msg)

    config = Config.from_yaml(config_path)
    if seed is not None:
        config.require_sft().seed = seed
    sft_finetune(
        config,
        pretrain_checkpoint=pretrain_checkpoint,
        resume_from=resume_from,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        profile_steps=profile_steps,
    )
