from pathlib import Path

import jax
import pytest
import yaml

from nanodiffusion.config import ModelConfig
from nanodiffusion.model.transformer import Transformer


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
            "max_steps": 100,
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p
