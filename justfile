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

schema:
    uv run nanodiffusion config gen-schema

tui *args:
    cd tui && cargo run --release -- {{ args }}

tui-build:
    cd tui && cargo build --release

tui-check:
    cd tui && cargo fmt --check && cargo clippy -- -D warnings

check: lint format typecheck test
