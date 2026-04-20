"""Session-scoped fixtures for the serve test suite.

The shared ``saved_checkpoint`` lives in ``tests/conftest.py`` so the
inference tests can reuse it without import gymnastics.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nanodiffusion.inference import Runtime, SampleConfigOverride, load_runtime
from nanodiffusion.serve import create_app


@pytest.fixture(scope="session")
def serve_runtime(saved_checkpoint: Path) -> Runtime:
    return load_runtime(saved_checkpoint, overrides=SampleConfigOverride())


@pytest.fixture(scope="session")
def serve_app(saved_checkpoint: Path) -> FastAPI:
    return create_app(checkpoint=saved_checkpoint, overrides=SampleConfigOverride())


@pytest.fixture(scope="session")
def client(serve_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(serve_app) as c:
        yield c
