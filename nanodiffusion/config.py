import math
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


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
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    warmup_steps: int = Field(default=2500, ge=0)
    max_steps: int = Field(default=100_000, gt=0)
    ema_decay: float = Field(default=0.9999, ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    log_every: int = Field(default=100, gt=0)
    save_every: int = Field(default=5000, gt=0)
    run_dir: Path = Path("runs/pretrain")

    @model_validator(mode="after")
    def _check_steps(self) -> "TrainConfig":
        if self.max_steps <= self.warmup_steps:
            msg = (
                f"max_steps ({self.max_steps}) must exceed warmup_steps "
                f"({self.warmup_steps}); otherwise the cosine schedule has "
                "no decay phase."
            )
            raise ValueError(msg)
        return self


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
    max_empty_passes: int = 100


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
