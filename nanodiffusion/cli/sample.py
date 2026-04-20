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
@click.option("--steps", default=None, type=int, help="Override sample.steps.")
@click.option(
    "--temperature", default=None, type=float, help="Override sample.temperature."
)
@click.option("--top-k", default=None, type=int, help="Override sample.top_k.")
@click.option("--top-p", default=None, type=float, help="Override sample.top_p.")
@click.option(
    "--max-length", default=None, type=int, help="Override sample.max_length."
)
@click.option("--seed", default=42, show_default=True, type=int)
def sample_command(
    *,
    checkpoint: Path,
    prompt: str,
    steps: int | None,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    max_length: int | None,
    seed: int,
) -> None:
    """Generate text via iterative unmasking."""
    import jax
    import jax.numpy as jnp

    from nanodiffusion import sampler
    from nanodiffusion.chat import render_for_completion
    from nanodiffusion.inference import SampleConfigOverride, load_runtime

    log = structlog.get_logger()

    runtime = load_runtime(
        checkpoint,
        overrides=SampleConfigOverride(
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
        ),
    )

    prompt_ids = render_for_completion(
        runtime.tok, {"messages": [{"role": "user", "content": prompt}]}
    )
    prompt_tokens = jnp.array(prompt_ids)
    defaults = runtime.defaults

    log.info(
        "sampling",
        steps=defaults.steps,
        max_length=defaults.max_length,
        prompt_len=len(prompt_ids),
    )

    tokens = sampler.sample_tokens(
        runtime.model,
        prompt_tokens,
        schedule=runtime.schedule,
        mask_token_id=runtime.tok.mask_token_id,
        max_length=defaults.max_length,
        steps=defaults.steps,
        temperature=defaults.temperature,
        top_k=defaults.top_k,
        top_p=defaults.top_p,
        key=jax.random.PRNGKey(seed),
    )

    click.echo(runtime.tok.decode(tokens.tolist()))
