"""Exercise generate_blocking and generate_stream against a real runtime."""

import pytest

from nanodiffusion.config import SampleConfig
from nanodiffusion.inference import Runtime
from nanodiffusion.serve.generation import generate_blocking, generate_stream
from nanodiffusion.serve.protocol import ChatRequest


def _request(steps: int = 4, max_length: int = 32, **overrides: object) -> ChatRequest:
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": "hi"}],
        "steps": steps,
        "max_length": max_length,
        "seed": 0,
    }
    payload.update(overrides)
    return ChatRequest.model_validate(payload)


def test_blocking_returns_filled_response(serve_runtime: Runtime) -> None:
    req = _request()
    res = generate_blocking(serve_runtime, req)
    assert len(res.tokens) == 32
    assert res.prompt_len < 32
    assert isinstance(res.text, str)


def test_stream_yields_steps_plus_one_frames(serve_runtime: Runtime) -> None:
    req = _request(steps=4, max_length=32)
    frames = list(generate_stream(serve_runtime, req))
    assert len(frames) == 5  # steps=4 yields 4 in-loop frames + 1 cleanup frame
    assert all(f.total == 4 for f in frames)
    assert frames[-1].step == 4
    assert all(len(f.tokens) == 32 for f in frames)


def test_stream_mask_positions_shrink_to_empty(serve_runtime: Runtime) -> None:
    frames = list(generate_stream(serve_runtime, _request()))
    assert frames[0].mask_positions, "first frame still has masks"
    assert frames[-1].mask_positions == [], "final frame has nothing masked"


def test_seed_is_deterministic(serve_runtime: Runtime) -> None:
    a = generate_blocking(serve_runtime, _request(seed=123))
    b = generate_blocking(serve_runtime, _request(seed=123))
    assert a.tokens == b.tokens


def test_max_length_above_model_max_seq_len_rejected(serve_runtime: Runtime) -> None:
    with pytest.raises(ValueError, match="exceeds model"):
        generate_blocking(serve_runtime, _request(max_length=10_000))


def test_empty_messages_rejected(serve_runtime: Runtime) -> None:
    req = ChatRequest(messages=[], steps=4, max_length=32, seed=0)
    with pytest.raises(ValueError, match="no messages"):
        generate_blocking(serve_runtime, req)


def test_bad_alternation_rejected(serve_runtime: Runtime) -> None:
    req = ChatRequest(
        messages=[{"role": "assistant", "content": "hi"}],
        steps=4,
        max_length=32,
        seed=0,
    )
    with pytest.raises(ValueError, match="alternate"):
        generate_blocking(serve_runtime, req)


def test_stream_validates_eagerly(serve_runtime: Runtime) -> None:
    """Invalid requests must raise from the call itself, not from
    iterating the returned generator — the SSE handler maps that to 422
    before committing to a stream."""
    with pytest.raises(ValueError, match="exceeds model"):
        generate_stream(serve_runtime, _request(max_length=10_000))


def _defaults(**overrides: object) -> SampleConfig:
    return SampleConfig().model_copy(update=overrides)


def test_with_overrides_falls_back_to_defaults_when_request_omits_fields() -> None:
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert _defaults().with_overrides(req) == _defaults()


def test_with_overrides_request_overrides_each_field() -> None:
    req = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        steps=8,
        temperature=0.5,
        top_k=10,
        top_p=0.9,
        max_length=128,
    )
    resolved = _defaults().with_overrides(req)
    assert resolved == _defaults(
        steps=8, temperature=0.5, top_k=10, top_p=0.9, max_length=128
    )


def test_with_overrides_top_k_zero_override_is_respected() -> None:
    """Regression: ``top_k=0`` is a valid request value; falsy guards must not
    treat it as "unset" and silently use the default."""
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], top_k=0)
    assert _defaults(top_k=10).with_overrides(req).top_k == 0


def test_runtime_defaults_reflect_session_overrides(serve_runtime: Runtime) -> None:
    """The session fixture builds a runtime with no overrides; defaults must
    therefore mirror the checkpointed ``config.sample`` (steps=4, max_length=32)."""
    assert serve_runtime.defaults.steps == 4
    assert serve_runtime.defaults.max_length == 32
