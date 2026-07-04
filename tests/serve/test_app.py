"""FastAPI integration tests via TestClient (HTTP + SSE)."""

import json

import structlog
from fastapi.testclient import TestClient


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": "hi"}],
        "steps": 4,
        "max_length": 32,
        "seed": 0,
    }
    payload.update(overrides)
    return payload


def _parse_sse(body: str) -> list[dict[str, object]]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_health_returns_expected_shape(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["train_step"] == 0
    assert body["max_seq_len"] == 64
    assert body["vocab_size"] > 50000
    assert body["sample_defaults"]["max_length"] == 32


def test_chat_returns_response(client: TestClient) -> None:
    res = client.post("/api/chat", json=_payload())
    assert res.status_code == 200
    body = res.json()
    assert len(body["tokens"]) == 32
    assert body["prompt_len"] < 32


def test_chat_rejects_oversized_max_length(client: TestClient) -> None:
    res = client.post("/api/chat", json=_payload(max_length=10_000))
    assert res.status_code == 422


def test_chat_rejects_bad_alternation(client: TestClient) -> None:
    res = client.post(
        "/api/chat",
        json=_payload(messages=[{"role": "assistant", "content": "hi"}]),
    )
    assert res.status_code == 422


def test_stream_yields_expected_frames(client: TestClient) -> None:
    res = client.post("/api/chat/stream", json=_payload(steps=4))
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(res.text)
    assert [f["step"] for f in frames] == [0, 1, 2, 3, 4]
    assert all(f["total"] == 4 for f in frames)
    assert frames[-1]["mask_positions"] == []


def test_stream_rejects_invalid_request(client: TestClient) -> None:
    res = client.post("/api/chat/stream", json=_payload(max_length=10_000))
    assert res.status_code == 422


def test_422_response_carries_detail(client: TestClient) -> None:
    res = client.post("/api/chat", json=_payload(max_length=10_000))
    assert res.status_code == 422
    assert "detail" in res.json()
    assert "max_length" in res.json()["detail"]


def test_successful_request_emits_access_log(client: TestClient) -> None:
    with structlog.testing.capture_logs() as entries:
        client.get("/api/health")
    access = [e for e in entries if e.get("event") == "request.complete"]
    assert len(access) == 1
    assert access[0]["status"] == 200
    assert access[0]["log_level"] == "info"
    assert isinstance(access[0]["duration_ms"], float)


def test_rejected_request_logs_reason_and_warning_access(client: TestClient) -> None:
    with structlog.testing.capture_logs() as entries:
        client.post("/api/chat", json=_payload(max_length=10_000))
    reject = next(e for e in entries if e.get("event") == "request.rejected")
    access = next(e for e in entries if e.get("event") == "request.complete")
    assert "max_length" in reject["detail"]
    assert access["status"] == 422
    assert access["log_level"] == "warning"
