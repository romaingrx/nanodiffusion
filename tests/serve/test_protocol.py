"""Pure Pydantic model tests — no runtime, no fixtures."""

import pytest
from pydantic import ValidationError

from nanodiffusion.serve.protocol import (
    EXPORTED_MODELS,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Message,
    SampleDefaults,
    StreamFrame,
    schema_document,
)


def test_message_requires_valid_role() -> None:
    Message(role="user", content="hi")
    Message(role="assistant", content="hi")
    Message(role="system", content="hi")
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "robot", "content": "hi"})


def test_chat_request_overrides_optional() -> None:
    req = ChatRequest(messages=[Message(role="user", content="hi")])
    assert req.steps is None
    assert req.temperature is None
    assert req.max_length is None
    assert req.seed is None


def test_chat_request_rejects_bad_values() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=[], steps=0)
    with pytest.raises(ValidationError):
        ChatRequest(messages=[], temperature=0.0)
    with pytest.raises(ValidationError):
        ChatRequest(messages=[], top_p=1.5)
    with pytest.raises(ValidationError):
        ChatRequest(messages=[], top_p=0.0)
    with pytest.raises(ValidationError):
        ChatRequest(messages=[], max_length=0)


def test_stream_frame_roundtrip() -> None:
    frame = StreamFrame(
        step=3, total=4, tokens=[1, 2, 3], text="hi", mask_positions=[1]
    )
    restored = StreamFrame.model_validate_json(frame.model_dump_json())
    assert restored == frame


def test_sample_defaults_must_be_positive() -> None:
    SampleDefaults(steps=4, temperature=1.0, top_k=0, top_p=1.0, max_length=32)
    with pytest.raises(ValidationError):
        SampleDefaults(steps=0, temperature=1.0, top_k=0, top_p=1.0, max_length=32)


def test_exported_models_covers_all_wire_types() -> None:
    expected = {
        Message,
        SampleDefaults,
        ChatRequest,
        ChatResponse,
        StreamFrame,
        HealthResponse,
    }
    assert set(EXPORTED_MODELS) == expected


def test_schema_document_has_every_exported_model() -> None:
    doc = schema_document()
    assert "$defs" in doc
    defs = doc["$defs"]
    assert isinstance(defs, dict)
    for model in EXPORTED_MODELS:
        assert model.__name__ in defs
