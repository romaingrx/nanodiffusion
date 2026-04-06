set dotenv-load

export UV_FROZEN := "1"

default:
    @just --list

test *args:
    uv run pytest {{ args }}

lint:
    uv run ruff check

format:
    uv run ruff format

typecheck:
    uv run basedpyright

check: lint format typecheck test
