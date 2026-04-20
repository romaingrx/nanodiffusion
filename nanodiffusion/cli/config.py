"""``nanodiffusion config ...`` commands: schema generation and validation."""

import json
from pathlib import Path

import click

from nanodiffusion import constants

_GENERATED_NOTE = (
    "Generated schema for nanodiffusion.config.Config. "
    "Do not hand-edit — regenerate with `just schema`."
)


def schema_document() -> dict[str, object]:
    """Pydantic JSON Schema for :class:`Config` plus a regenerate-me note.

    Keeps :meth:`Config.model_json_schema` pristine for programmatic
    callers and confines the on-disk artifact tweak (the regenerate-me
    note) to this module. Any existing ``description`` — e.g. from a
    ``Config`` class docstring — is preserved and the note is appended
    so neither message is lost.
    """
    from nanodiffusion.config import Config

    schema: dict[str, object] = Config.model_json_schema()
    existing = schema.get("description")
    if isinstance(existing, str) and existing:
        schema["description"] = f"{existing}\n\n{_GENERATED_NOTE}"
    else:
        schema["description"] = _GENERATED_NOTE
    return schema


@click.group(name="config")
def config_group() -> None:
    """Config schema and validation commands."""


@config_group.command(name="gen-schema")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=constants.SCHEMA_PATH,
    show_default=True,
    help="Destination for the JSON Schema file.",
)
def gen_schema(output: Path) -> None:
    """Regenerate the JSON Schema for the top-level :class:`Config`.

    The schema is derived from the pydantic models so it always
    reflects the live Python definition. Commit the regenerated file
    whenever ``nanodiffusion/config.py`` changes.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema_document(), indent=2) + "\n")
    click.echo(f"Wrote schema to {output}")


@config_group.command(name="validate")
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def validate(paths: tuple[Path, ...]) -> None:
    """Validate one or more YAML config files against the :class:`Config` schema.

    Runs the full pydantic validator stack — field constraints and
    ``@model_validator`` hooks alike — and reports every failure
    before exiting non-zero. Use directly from the command line or
    wire into a pre-commit hook to catch drift at commit time.
    """
    from pydantic import ValidationError

    from nanodiffusion.config import Config

    failures: list[tuple[Path, str]] = []
    for path in paths:
        try:
            Config.from_yaml(path)
        except ValidationError as exc:
            failures.append((path, str(exc)))
        else:
            click.echo(f"ok: {path}")

    if failures:
        for path, err in failures:
            click.echo(f"fail: {path}\n{err}", err=True)
        raise click.exceptions.Exit(1)
