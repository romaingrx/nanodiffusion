"""Nanodiffusion CLI entry point."""

import logging

import click
import structlog


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def main(*, verbose: bool) -> None:
    """Nanodiffusion command-line interface."""
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


from nanodiffusion.cli.data import data_group  # noqa: E402
from nanodiffusion.cli.sample import sample_command  # noqa: E402

main.add_command(sample_command)
main.add_command(data_group)
