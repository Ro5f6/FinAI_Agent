"""Tests for the HTTP API."""

from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from finai.agent.session import app_version
from finai.api.deps import get_graph
from finai.api.main import app, create_app
from finai.api.schemas import ChatRequest, StreamEvent
from finai.config import settings


class _ScriptedGraph:
    """A graph factory for tests: no Redis, no model; counts what it does."""

    def __init__(self) -> None:
        self.built = 0
        self.closed = 0
        self.graph = object()

    def __call__(self):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def factory():
            self.built += 1
            try:
                yield self.graph
            finally:
                self.closed += 1

        return factory()


def test_the_app_assembles_with_the_chat_router() -> None:
    created = create_app(_ScriptedGraph())
    assert isinstance(created, FastAPI)
    assert created.version == app_version()
    assert isinstance(app, FastAPI)


def test_the_openapi_schema_is_served() -> None:
    response = TestClient(create_app(_ScriptedGraph())).get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "FinAI-Agent"


# --- schemas (2.1.3) ------------------------------------------------------------


def test_a_request_needs_only_a_message() -> None:
    request = ChatRequest(message="  What's NVDA trading at?  ")
    assert request.message == "What's NVDA trading at?"  # stripped
    assert request.session_id is None


@pytest.mark.parametrize(
    "message", ["", "   ", "x" * (settings.CHAT_MAX_MESSAGE_CHARS + 1)]
)
def test_blank_or_too_long_messages_are_rejected(message: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message=message)


def test_the_longest_allowed_message_is_accepted() -> None:
    assert ChatRequest(message="x" * settings.CHAT_MAX_MESSAGE_CHARS)


@pytest.mark.parametrize("session_id", ["abc", "3f2b9c1e-portfolio_2", "a" * 64])
def test_valid_session_ids(session_id: str) -> None:
    assert ChatRequest(message="q", session_id=session_id).session_id == session_id


@pytest.mark.parametrize(
    "session_id", ["", "a" * 65, "user:other", "../etc", "a b", "thread*", "ümlaut"]
)
def test_session_ids_that_could_misuse_a_storage_key_are_rejected(
    session_id: str,
) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="q", session_id=session_id)


def test_a_request_cannot_name_its_user() -> None:
    """user_id comes from authentication only; one in the body is an error."""
    with pytest.raises(ValidationError, match="user_id"):
        ChatRequest.model_validate({"message": "q", "user_id": "someone-else"})


def test_an_event_encodes_as_named_sse_with_json_data() -> None:
    event = StreamEvent(
        type="tool_start",
        data={"tool": "get_live_market_data", "args": {"ticker": "NVDA"}},
    )
    assert event.to_sse().encode() == (
        b"event: tool_start\r\n"
        b'data: {"tool":"get_live_market_data","args":{"ticker":"NVDA"}}\r\n\r\n'
    )


def test_text_with_newlines_stays_on_one_data_line() -> None:
    encoded = (
        StreamEvent(type="token", data={"text": "line one\nline two"}).to_sse().encode()
    )
    assert encoded.count(b"data: ") == 1
    assert b"line one\\nline two" in encoded


def test_unknown_event_types_are_rejected() -> None:
    with pytest.raises(ValidationError):
        StreamEvent(type="debug", data={})


# --- the graph, built once at startup (2.1.4) -------------------------------------


def test_the_graph_is_built_once_and_shared_by_every_request() -> None:
    graphs = _ScriptedGraph()
    app = create_app(graphs)
    seen = []

    @app.get("/probe")
    def probe(graph: Annotated[object, Depends(get_graph)]) -> dict:
        seen.append(graph)
        return {}

    with TestClient(app) as client:  # runs the lifespan
        assert graphs.built == 1
        for _ in range(3):
            assert client.get("/probe").status_code == 200
        assert graphs.closed == 0  # open while serving
    assert seen == [graphs.graph] * 3
    assert (graphs.built, graphs.closed) == (1, 1)  # closed at shutdown


# --- health probes (2.1.5) ----------------------------------------------------------

import time

from finai.api.routers import health


def _client() -> TestClient:
    return TestClient(create_app(_ScriptedGraph()))


def _fail(error: Exception):
    def check() -> None:
        raise error

    return check


def test_healthz_checks_nothing_else(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "CHECKS", {"qdrant": _fail(AssertionError("called"))})
    response = _client().get("/healthz")
    assert (response.status_code, response.json()) == (200, {"status": "ok"})


def test_readyz_when_both_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health, "CHECKS", {"qdrant": lambda: None, "redis": lambda: None}
    )
    response = _client().get("/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"qdrant": "ok", "redis": "ok"},
    }


def test_readyz_is_503_and_names_the_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health,
        "CHECKS",
        {"qdrant": lambda: None, "redis": _fail(ConnectionError("refused"))},
    )
    response = _client().get("/readyz")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"qdrant": "ok", "redis": "unreachable: ConnectionError"},
    }
    assert "refused" not in response.text  # no internal detail


def test_a_hung_dependency_fails_the_probe_quickly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe answers at the timeout. The stuck check's thread can't be
    stopped, so it is timed inside the event loop, then released."""
    import asyncio
    import threading

    from fastapi import Response

    release = threading.Event()
    monkeypatch.setattr(health, "CHECK_TIMEOUT_S", 0.2)
    monkeypatch.setattr(
        health, "CHECKS", {"qdrant": lambda: release.wait(10), "redis": lambda: None}
    )

    async def probe():
        response = Response()
        started = time.monotonic()
        body = await health.readyz(response)
        elapsed = time.monotonic() - started
        release.set()
        return response, body, elapsed

    response, body, elapsed = asyncio.run(probe())

    assert elapsed < 0.5
    assert response.status_code == 503
    assert body["checks"]["qdrant"] == "unreachable: no answer in 0.2 s"


def test_probes_need_no_api_key() -> None:
    """Kubernetes probes don't authenticate (the chat endpoint will, 2.1.9)."""
    assert _client().get("/healthz", headers={}).status_code == 200


@pytest.mark.qdrant
@pytest.mark.redis
def test_live_readyz_with_compose_services() -> None:
    response = TestClient(create_app(_ScriptedGraph())).get("/readyz")
    assert response.status_code == 200, response.json()
    assert response.json()["status"] == "ready"
