"""Session-scoped fixtures for the serve test suite.

Building a real-vocab model and round-tripping a checkpoint through
``save_checkpoint`` costs a few seconds (plus JIT warmup), so every
fixture here is ``scope="session"`` — the whole suite shares one
loaded runtime and one live ``TestClient``.
"""

from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import optax
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nanodiffusion.checkpoint import save_checkpoint, write_config
from nanodiffusion.config import Config, ModelConfig
from nanodiffusion.model import Transformer
from nanodiffusion.serve import (
    Runtime,
    SampleDefaultsOverride,
    create_app,
    load_runtime,
)
from nanodiffusion.tokenizer import Tokenizer


@pytest.fixture(scope="session")
def _serve_tok() -> Tokenizer:
    return Tokenizer()


@pytest.fixture(scope="session")
def saved_checkpoint(
    tmp_path_factory: pytest.TempPathFactory, _serve_tok: Tokenizer
) -> Path:
    """Write a tiny real-vocab checkpoint once per session."""
    tmp = tmp_path_factory.mktemp("serve")
    config = Config(
        model=ModelConfig(
            vocab_size=_serve_tok.vocab_size,
            num_layers=2,
            hidden_dim=64,
            num_heads=4,
            max_seq_len=64,
        ),
        sample=Config().sample.model_copy(update={"steps": 4, "max_length": 32}),
    )
    key = jax.random.PRNGKey(0)
    model = Transformer(config.model, key=key)
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ckpt = tmp / "step_0"
    save_checkpoint(
        ckpt,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        step=0,
        cursor=None,
    )
    write_config(ckpt, config)
    return ckpt


@pytest.fixture(scope="session")
def serve_runtime(saved_checkpoint: Path) -> Runtime:
    return load_runtime(saved_checkpoint, overrides=SampleDefaultsOverride())


@pytest.fixture(scope="session")
def serve_app(saved_checkpoint: Path) -> FastAPI:
    return create_app(checkpoint=saved_checkpoint, overrides=SampleDefaultsOverride())


@pytest.fixture(scope="session")
def client(serve_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(serve_app) as c:
        yield c
