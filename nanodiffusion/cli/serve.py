"""``nanodiffusion serve`` and ``nanodiffusion schema`` commands."""

from pathlib import Path

import click


@click.command(name="serve")
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--steps", default=None, type=int, help="Override sample.steps.")
@click.option(
    "--temperature", default=None, type=float, help="Override sample.temperature."
)
@click.option("--top-k", default=None, type=int, help="Override sample.top_k.")
@click.option("--top-p", default=None, type=float, help="Override sample.top_p.")
@click.option(
    "--max-length", default=None, type=int, help="Override sample.max_length."
)
def serve_command(
    *,
    checkpoint: Path,
    host: str,
    port: int,
    steps: int | None,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    max_length: int | None,
) -> None:
    """Serve a trained checkpoint over HTTP + SSE.

    Single-tenant by design: one JAX model per process, no concurrency,
    no auth. Binding ``--host 0.0.0.0`` exposes the endpoint to anything
    on the network; only do so behind a trusted reverse proxy.
    """
    import uvicorn

    from nanodiffusion.inference import SampleConfigOverride
    from nanodiffusion.serve import create_app

    app = create_app(
        checkpoint=checkpoint,
        overrides=SampleConfigOverride(
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
        ),
    )
    # log_config=None keeps uvicorn from replacing our structlog-backed root
    # handler; access/error logs inherit from the root logger instead.
    uvicorn.run(app, host=host, port=port, log_config=None)


@click.command(name="schema")
@click.option(
    "--output",
    default=Path("schemas/protocol.json"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
def schema_command(*, output: Path) -> None:
    """Regenerate the wire-protocol JSON Schema from Pydantic models.

    Commit the regenerated file whenever ``nanodiffusion/serve/protocol.py``
    changes. The Rust TUI consumes this via ``typify::import_types!``.
    """
    from nanodiffusion.serve.protocol import dump_schema

    dump_schema(output)
    click.echo(f"Wrote schema to {output}")
