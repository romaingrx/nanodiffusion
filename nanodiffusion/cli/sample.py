"""``nanodiffusion sample`` command: generate text via iterative unmasking."""

from pathlib import Path

import click
import structlog


@click.command(name="sample")
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--prompt", required=True, type=str)
@click.option("--steps", default=64, show_default=True, type=int)
@click.option("--temperature", default=1.0, show_default=True, type=float)
@click.option("--top-k", default=0, show_default=True, type=int)
@click.option("--top-p", default=1.0, show_default=True, type=float)
@click.option("--max-length", default=256, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
def sample_command(
    *,
    checkpoint: Path,
    prompt: str,
    steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_length: int,
    seed: int,
) -> None:
    """Generate text via iterative unmasking."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from nanodiffusion import sampler  # noqa: PLC0415
    from nanodiffusion.chat import render_for_completion  # noqa: PLC0415
    from nanodiffusion.checkpoint import load_model  # noqa: PLC0415
    from nanodiffusion.config import Config  # noqa: PLC0415
    from nanodiffusion.constants import CONFIG_SIDECAR_FILENAME  # noqa: PLC0415
    from nanodiffusion.model import Transformer  # noqa: PLC0415
    from nanodiffusion.schedule import LogLinearSchedule  # noqa: PLC0415
    from nanodiffusion.tokenizer import Tokenizer  # noqa: PLC0415

    log = structlog.get_logger()

    config = Config.from_yaml(checkpoint / CONFIG_SIDECAR_FILENAME)
    tok = Tokenizer()
    schedule = LogLinearSchedule()

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model_skeleton = Transformer(config.model, key=model_key)
    model = load_model(checkpoint, model_skeleton=model_skeleton, which="ema")

    prompt_ids = render_for_completion(
        tok, {"messages": [{"role": "user", "content": prompt}]}
    )
    prompt_tokens = jnp.array(prompt_ids)

    log.info(
        "sampling",
        steps=steps,
        max_length=max_length,
        prompt_len=len(prompt_ids),
    )

    tokens = sampler.sample_tokens(
        model,
        prompt_tokens,
        schedule=schedule,
        mask_token_id=tok.mask_token_id,
        max_length=max_length,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        key=key,
    )

    click.echo(tok.decode(tokens.tolist()))
