import math
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import jax.numpy as jnp
import yaml
from pydantic import BaseModel, Field, model_validator


@runtime_checkable
class OptimizerHyperparams(Protocol):
    """Subset of training config fields that ``make_optimizer`` reads.

    Typing the optimizer builder against this Protocol lets both
    :class:`TrainConfig` and :class:`SFTConfig` satisfy it structurally,
    so neither has to inherit from the other nor duplicate the optimizer
    fields on a shared base class.
    """

    learning_rate: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    grad_clip: float


def _round_to_multiple(x: int, multiple: int) -> int:
    return math.ceil(x / multiple) * multiple


class ModelConfig(BaseModel):
    vocab_size: int = 50264  # GPT-2 (50257) + 7 special tokens (see SpecialToken enum)
    num_layers: int = 12
    hidden_dim: int = 768
    num_heads: int = 12
    max_seq_len: int = 1024
    ffn_mult: int = 4
    compute_dtype: Literal["float32", "bfloat16"] = "float32"
    remat_policy: Literal["none", "nothing", "dots_no_batch", "dots", "everything"] = (
        "none"
    )

    @property
    def jnp_dtype(self) -> type:
        """Resolve :attr:`compute_dtype` to the JAX scalar type used at
        trace time (``jnp.float32`` or ``jnp.bfloat16``)."""
        return jnp.float32 if self.compute_dtype == "float32" else jnp.bfloat16

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def ffn_dim(self) -> int:
        return _round_to_multiple(int(2 / 3 * self.ffn_mult * self.hidden_dim), 256)

    @model_validator(mode="after")
    def _check_attention_shape(self) -> "ModelConfig":
        if self.hidden_dim % self.num_heads != 0:
            msg = (
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
            raise ValueError(msg)
        return self


class TrainConfig(BaseModel):
    seed: int = 42
    batch_size: int = Field(default=32, gt=0)
    grad_accum_steps: int = Field(default=1, ge=1)
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
        if self.batch_size % self.grad_accum_steps != 0:
            msg = (
                f"batch_size ({self.batch_size}) must be divisible by "
                f"grad_accum_steps ({self.grad_accum_steps})"
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


class SFTDatasetConfig(BaseModel):
    """One entry in an SFT task mixture.

    ``name`` must resolve via the chat-dataset registry at run time;
    ``epochs`` repeats the same source ``epochs`` times inside
    :class:`TaskMixture` to implement nanochat-style oversampling.
    """

    name: str
    epochs: int = Field(default=1, gt=0)


def _default_sft_datasets() -> list[SFTDatasetConfig]:
    """Mirrors nanochat's SFT mixture scoped to ROM-17's three datasets.

    GSM8K and identity are oversampled relative to SmolTalk — small
    datasets drowning in a bigger chat corpus need the extra passes to
    leave any trace in the final model. Matches nanochat's
    ``chat_sft.py:train_tasks`` shape.
    """
    return [
        SFTDatasetConfig(name="smoltalk"),
        SFTDatasetConfig(name="gsm8k", epochs=4),
        SFTDatasetConfig(name="identity", epochs=2),
    ]


class SFTConfig(BaseModel):
    seed: int = 42
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    warmup_steps: int = Field(default=50, ge=0)
    max_steps: int = Field(default=2000, gt=0)
    ema_decay: float = Field(default=0.999, ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    log_every: int = Field(default=20, gt=0)
    save_every: int = Field(default=500, gt=0)
    run_dir: Path = Path("runs/sft")
    datasets: list[SFTDatasetConfig] = Field(default_factory=_default_sft_datasets)
    tokenizer_batch_size: int = 128
    tokenizer_threads: int = 4
    prefetch_size: int = 2
    max_empty_passes: int = 100

    @model_validator(mode="after")
    def _check_steps(self) -> "SFTConfig":
        if self.max_steps <= self.warmup_steps:
            msg = (
                f"max_steps ({self.max_steps}) must exceed warmup_steps "
                f"({self.warmup_steps}); otherwise the cosine schedule has "
                "no decay phase."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_datasets(self) -> "SFTConfig":
        if not self.datasets:
            err = "SFTConfig.datasets must contain at least one entry"
            raise ValueError(err)
        return self


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    sample: SampleConfig = SampleConfig()
    data: DataConfig = DataConfig()
    sft: SFTConfig = SFTConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with Path(path).open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
