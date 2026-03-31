from pathlib import Path

from nanodiffusion.config import Config, ModelConfig, SampleConfig, TrainConfig
from nanodiffusion.types import Mask, MaskBatch, TokenBatch, Tokens


def test_import_types() -> None:
    assert Tokens is not None
    assert TokenBatch is not None
    assert Mask is not None
    assert MaskBatch is not None


def test_default_config() -> None:
    cfg = Config()
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.train, TrainConfig)
    assert isinstance(cfg.sample, SampleConfig)


def test_model_config_head_dim() -> None:
    cfg = ModelConfig(hidden_dim=768, num_heads=12)
    assert cfg.head_dim == 64


def test_config_from_yaml(config_path: Path) -> None:
    cfg = Config.from_yaml(config_path)
    assert cfg.model.num_layers == 4
    assert cfg.model.hidden_dim == 256
    assert cfg.train.batch_size == 8
    assert cfg.train.max_steps == 100
    # Defaults preserved for unset fields
    assert cfg.sample.steps == 64
