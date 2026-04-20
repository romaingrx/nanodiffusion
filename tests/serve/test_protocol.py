"""Pure Pydantic model tests — no runtime, no fixtures."""

import pytest
from pydantic import ValidationError

from nanodiffusion.config import SampleConfig
from nanodiffusion.serve.protocol import (
    EXPORTED_MODELS,
    ChatRequest,
    ChatResponse,
    FrozenBaseModel,
    HealthResponse,
    StreamFrame,
    schema_document,
)


def test_chat_request_accepts_typed_message_dicts() -> None:
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert req.messages[0]["role"] == "user"


def test_chat_request_rejects_invalid_role() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "robot", "content": "hi"}])


def test_chat_request_overrides_optional() -> None:
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
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


def test_chat_request_mirrors_sample_config_fields() -> None:
    """``ChatRequest`` must have a nullable variant of every ``SampleConfig`` field
    so the resolve helper can override every default."""
    assert SampleConfig.model_fields.keys() <= ChatRequest.model_fields.keys()


def test_frozen_base_model_rejects_mutation() -> None:
    """Every wire model is frozen via :class:`FrozenBaseModel`."""
    res = ChatResponse(text="hi", tokens=[1], prompt_len=0)
    assert isinstance(res, FrozenBaseModel)
    with pytest.raises(ValidationError):
        res.text = "bye"  # pyright: ignore[reportAttributeAccessIssue]


def test_exported_models_covers_all_wire_types() -> None:
    expected = {ChatRequest, ChatResponse, StreamFrame, HealthResponse}
    assert set(EXPORTED_MODELS) == expected


def test_schema_document_has_every_exported_model() -> None:
    doc = schema_document()
    assert "$defs" in doc
    defs = doc["$defs"]
    assert isinstance(defs, dict)
    for model in EXPORTED_MODELS:
        assert model.__name__ in defs
