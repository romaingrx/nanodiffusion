import logging
from pathlib import Path

import click
import structlog


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def main(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
    )


@main.command()
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
def sample(
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
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from nanodiffusion import sampler  # noqa: PLC0415
    from nanodiffusion.chat import render_for_completion  # noqa: PLC0415
    from nanodiffusion.config import Config  # noqa: PLC0415
    from nanodiffusion.model.transformer import Transformer  # noqa: PLC0415
    from nanodiffusion.schedule import LogLinearSchedule  # noqa: PLC0415
    from nanodiffusion.tokenizer import Tokenizer  # noqa: PLC0415

    log = structlog.get_logger()

    config = Config.from_yaml(checkpoint / "config.yaml")
    tok = Tokenizer()
    schedule = LogLinearSchedule()

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    skeleton = Transformer(config.model, key=model_key)
    loaded: Transformer = eqx.tree_deserialise_leaves(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        checkpoint / "model.eqx", skeleton
    )
    jit_model = eqx.filter_jit(loaded)  # pyright: ignore[reportUnknownArgumentType]

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
        jit_model,
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
