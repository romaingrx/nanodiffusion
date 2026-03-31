import logging

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
