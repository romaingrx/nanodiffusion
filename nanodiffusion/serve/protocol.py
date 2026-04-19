"""Wire protocol for the serve layer — single source of truth.

Pydantic models here are the only place the HTTP/WebSocket JSON shape is
defined. :func:`dump_schema` exports them as one JSON Schema file so a
downstream Rust TUI (or any other client) can regenerate typed bindings
with ``typify::import_types!`` whenever this module changes.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import models_json_schema

from nanodiffusion.chat import Role


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class SampleDefaults(BaseModel):
    """Resolved sampling hyperparameters a request falls back to."""

    model_config = ConfigDict(frozen=True)

    steps: int = Field(gt=0)
    temperature: float = Field(gt=0)
    top_k: int = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_length: int = Field(gt=0)


class ChatRequest(BaseModel):
    messages: list[Message]
    steps: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, gt=0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_length: int | None = Field(default=None, gt=0)
    seed: int | None = None


class ChatResponse(BaseModel):
    text: str
    tokens: list[int]
    prompt_len: int


class StreamFrame(BaseModel):
    """One frame per unmasking step. ``step == total`` marks the final cleanup."""

    step: int = Field(ge=0)
    total: int = Field(gt=0)
    tokens: list[int]
    text: str
    mask_positions: list[int]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    checkpoint: str
    train_step: int = Field(ge=0)
    max_seq_len: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    sample_defaults: SampleDefaults


EXPORTED_MODELS: tuple[type[BaseModel], ...] = (
    Message,
    SampleDefaults,
    ChatRequest,
    ChatResponse,
    StreamFrame,
    HealthResponse,
)


def schema_document() -> dict[str, object]:
    """Combined JSON Schema with one ``$defs`` entry per exported model."""
    _, schema = models_json_schema(
        [(m, "validation") for m in EXPORTED_MODELS],
        title="NanodiffusionProtocol",
    )
    return schema


def dump_schema(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema_document(), indent=2) + "\n")
