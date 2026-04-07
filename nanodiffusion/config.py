import math
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


def _round_to_multiple(x: int, multiple: int) -> int:
    return math.ceil(x / multiple) * multiple


class ModelConfig(BaseModel):
    vocab_size: int = 50264  # GPT-2 (50257) + 7 special tokens (see SpecialToken enum)
    num_layers: int = 12
    hidden_dim: int = 768
    num_heads: int = 12
    max_seq_len: int = 1024
    dropout_rate: float = 0.0
    ffn_mult: int = 4

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def ffn_dim(self) -> int:
        return _round_to_multiple(int(2 / 3 * self.ffn_mult * self.hidden_dim), 256)


class TrainConfig(BaseModel):
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 2500
    max_steps: int = 100_000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    log_every: int = 100
    save_every: int = 5000
    eval_every: int = 1000


class SampleConfig(BaseModel):
    steps: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_length: int = 256


class DataConfig(BaseModel):
    dataset: str = "climbmix-400b"
    data_dir: Path = Path("data")
    num_train_shards: int | None = None
    tokenizer_batch_size: int = 128
    tokenizer_threads: int = 4
    prefetch_size: int = 4

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, v: str) -> str:
        # Lazy import keeps `from nanodiffusion.config import Config` cheap
        # for callers that never touch the data layer (e.g. sampling-only).
        from nanodiffusion.data.datasets import DATASETS  # noqa: PLC0415

        if v not in DATASETS:
            available = ", ".join(sorted(DATASETS)) or "(none)"
            msg = f"Unknown dataset {v!r}. Known: {available}"
            raise ValueError(msg)
        return v


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    sample: SampleConfig = SampleConfig()
    data: DataConfig = DataConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with Path(path).open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
