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
@click.option(
    "--steps",
    default=None,
    type=int,
    help="Override config.sample.steps for every request.",
)
@click.option(
    "--temperature",
    default=None,
    type=float,
    help="Override config.sample.temperature for every request.",
)
def serve_command(
    *,
    checkpoint: Path,
    host: str,
    port: int,
    steps: int | None,
    temperature: float | None,
) -> None:
    """Serve a trained checkpoint over HTTP/WebSocket.

    Single-tenant by design: one JAX model per process, no concurrency.
    """
    import uvicorn  # noqa: PLC0415

    from nanodiffusion.serve import SampleDefaultsOverride, create_app  # noqa: PLC0415

    app = create_app(
        checkpoint=checkpoint,
        overrides=SampleDefaultsOverride(steps=steps, temperature=temperature),
    )
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
    from nanodiffusion.serve.protocol import dump_schema  # noqa: PLC0415

    dump_schema(output)
    click.echo(f"Wrote schema to {output}")
