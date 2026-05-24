from pathlib import Path

import equinox as eqx
import jax
import optax
import pytest
import yaml
from jaxtyping import install_import_hook

install_import_hook("nanodiffusion", "beartype.beartype")

from nanodiffusion.checkpoint import (  # noqa: E402
    flush,
    make_manager,
    save_checkpoint,
    write_config,
)
from nanodiffusion.config import Config, ModelConfig  # noqa: E402
from nanodiffusion.model.transformer import Transformer  # noqa: E402
from nanodiffusion.tokenizer import Tokenizer  # noqa: E402


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        num_layers=2,
        hidden_dim=64,
        num_heads=4,
        max_seq_len=32,
    )


@pytest.fixture
def key() -> jax.Array:
    return jax.random.PRNGKey(0)


@pytest.fixture
def model(small_config: ModelConfig, key: jax.Array) -> Transformer:
    model_key, _ = jax.random.split(key)
    return Transformer(small_config, key=model_key)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    data = {
        "model": {
            "num_layers": 4,
            "hidden_dim": 256,
            "num_heads": 4,
        },
        "train": {
            "batch_size": 8,
            "warmup_steps": 10,
            "max_steps": 100,
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture
def tok() -> Tokenizer:
    """Shared tokenizer instance for tests that touch the data pipeline."""
    return Tokenizer()


@pytest.fixture(scope="session")
def saved_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny real-vocab checkpoint, written once per session.

    Shared by ``tests/test_inference.py`` and ``tests/serve/`` since both
    need a load-able checkpoint with a real-tokenizer vocab.
    """
    run_dir = tmp_path_factory.mktemp("inference")
    serve_tok = Tokenizer()
    config = Config(
        model=ModelConfig(
            vocab_size=serve_tok.vocab_size,
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

    write_config(run_dir, config)
    mngr = make_manager(run_dir)
    save_checkpoint(
        mngr,
        0,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.key(0),
        cursor=None,
    )
    flush(mngr)
    return run_dir
