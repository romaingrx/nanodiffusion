"""Nanodiffusion CLI entry point."""

import logging

import click

from nanodiffusion.logs import LogFormat
from nanodiffusion.logs import configure as configure_logging


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option(
    "--log-format",
    type=click.Choice(LogFormat),
    default=LogFormat.console,
    show_default=True,
    help="Log rendering: console for dev, json for log shippers.",
)
@click.option(
    "--external-logs/--no-external-logs",
    default=True,
    help="Keep or silence stdlib logs from third-party libs (uvicorn, fastapi, httpx).",
)
def main(*, verbose: bool, log_format: LogFormat, external_logs: bool) -> None:
    """Nanodiffusion command-line interface."""
    configure_logging(
        level=logging.DEBUG if verbose else logging.INFO,
        fmt=log_format,
        external_logs=external_logs,
    )


from nanodiffusion.cli.config import config_group  # noqa: E402
from nanodiffusion.cli.data import data_group  # noqa: E402
from nanodiffusion.cli.pretrain import pretrain_command  # noqa: E402
from nanodiffusion.cli.sample import sample_command  # noqa: E402
from nanodiffusion.cli.serve import schema_command, serve_command  # noqa: E402
from nanodiffusion.cli.sft import sft_command  # noqa: E402

main.add_command(sample_command)
main.add_command(data_group)
main.add_command(pretrain_command)
main.add_command(sft_command)
main.add_command(config_group)
main.add_command(serve_command)
main.add_command(schema_command)
