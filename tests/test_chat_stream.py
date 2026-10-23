"""The chat endpoint's event stream (2.1.6), end to end through the app.

The graph is the real one with a scripted planner and a stubbed quote, so
the planner, tools and verifier nodes all run; no model, no network.
"""

import datetime as dt
import json
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from finai.agent.graph import build_graph
from finai.api.main import create_app
from finai.api.streaming import pieces
from finai.tools import market_data

RIGHT = "NVDA is at $212.21, down 4.4% today. [Finnhub quote, 2026-10-22 20:00 UTC]"
WRONG = "NVDA is at $221.21."


TEST_KEY = "test-key-0123456789abcdef"
AUTH = {"X-API-Key": TEST_KEY}


@pytest.fixture(autouse=True)
def api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The demo user's key, sent by every client below (2.1.9)."""
    from pydantic import SecretStr

    from finai.config import settings

    monkeypatch.setattr(
        settings, "API_KEYS", SecretStr(f"demo:{TEST_KEY},bob:bob-key-0123456789abcd")
    )


@pytest.fixture(autouse=True)
def quote(monkeypatch: pytest.MonkeyPatch) -> None:
    data = market_data.MarketData(
        ticker="NVDA", name="NVIDIA", price=212.21, price_currency="USD", change_pct=-4.4,
        market_cap=None, market_cap_currency=None, week_52_high=None, week_52_low=None,
        pe_ttm=None, pe_annual=None, pe_source=None, eps_ttm=None, eps_ttm_period_end=None,
        eps_annual=None, eps_annual_period_end=None, eps_currency=None, eps_per=None,
        quote_time=dt.datetime(2026, 10, 22, 20, 0, tzinfo=dt.UTC),
    )  # fmt: skip
    monkeypatch.setattr(market_data, "get_live_market_data", lambda ticker: data)


def _client(wrong_drafts: int = 0) -> TestClient:
    drafts: list[str] = []

    def planner(messages, config=None) -> AIMessage:
        if not any(getattr(m, "tool_calls", None) for m in messages):
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "get_live_market_data",
                        "args": {"ticker": "NVDA"},
                    }
                ],
            )
        draft = WRONG if len(drafts) < wrong_drafts else RIGHT
        drafts.append(draft)
        return AIMessage(draft)

    @asynccontextmanager
    async def graph():
        yield build_graph(
            RunnableLambda(planner), checkpointer=InMemorySaver(), critic=None
        )

    return TestClient(create_app(graph), headers=AUTH)


def _events(body: str) -> list[tuple[str, dict]]:
    """The (event, data) pairs of an SSE body."""
    events = []
    for frame in body.replace("\r\n", "\n").split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line)
        if "event" in lines:
            events.append((lines["event"], json.loads(lines["data"])))
    return events


def _ask(client: TestClient, **body) -> list[tuple[str, dict]]:
    with client:
        response = client.post(
            "/api/v1/chat", json={"message": "What's NVDA at?", **body}
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return _events(response.text)


def test_progress_then_the_verified_answer() -> None:
    events = _ask(_client())
    kinds = [kind for kind, _ in events]

    assert kinds[:3] == ["tool_start", "tool_end", "status"]
    assert events[0][1] == {
        "tool": "get_live_market_data",
        "args": {"ticker": "NVDA"},
        "call_id": "c1",
    }
    assert events[1][1] == {
        "tool": "get_live_market_data",
        "call_id": "c1",
        "status": "success",
        "summary": "NVDA at 212.21 USD",
    }
    assert events[2][1] == {"stage": "verifying"}
    assert set(kinds[3:-1]) == {"token"}
    assert "".join(data["text"] for kind, data in events if kind == "token") == RIGHT
    kind, done = events[-1]
    assert kind == "done"
    assert (done["verdict"], done["revisions"]) == ("pass", 0)
    assert done["session_id"]


def test_a_rejected_draft_is_never_sent() -> None:
    """The planted wrong price: the stream says it is revising, never shows it."""
    events = _ask(_client(wrong_drafts=1))
    statuses = [data["stage"] for kind, data in events if kind == "status"]

    assert statuses == ["verifying", "revising", "verifying"]
    sent = json.dumps(events)
    assert "221.21" not in sent
    assert "".join(data["text"] for kind, data in events if kind == "token") == RIGHT
    assert events[-1][1]["revisions"] == 1


def test_a_given_session_is_continued() -> None:
    events = _ask(_client(), session_id="my-session")
    assert events[-1][1]["session_id"] == "my-session"


def test_an_invalid_request_is_rejected_before_streaming() -> None:
    with _client() as client:
        response = client.post("/api/v1/chat", json={"message": "", "user_id": "x"})
    assert response.status_code == 422


def test_pieces_rejoin_to_the_text_and_keep_their_size() -> None:
    text = "A long answer " * 20 + "with a final word."
    parts = pieces(text, size=30)
    assert "".join(parts) == text
    assert all(len(p) <= 30 + 15 for p in parts)  # a piece ends at the next word
    assert pieces("") == []


# --- sessions are per user (2.1.7) ---------------------------------------------------

from finai.api.deps import current_user


def test_a_session_id_continues_only_its_own_users_conversation() -> None:
    """Bob sending Alice's session ID starts his own conversation."""
    client = _client()
    app = client.app
    asked: list[int] = []

    def as_user(user: str):
        app.dependency_overrides[current_user] = lambda: user

    with client:
        graph = app.state.graph
        for user in ("alice", "alice", "bob"):
            as_user(user)
            client.post(
                "/api/v1/chat", json={"message": f"{user} asks", "session_id": "shared"}
            )
        alice = graph.get_state({"configurable": {"thread_id": "alice:shared"}}).values
        bob = graph.get_state({"configurable": {"thread_id": "bob:shared"}}).values
        asked.append(sum(1 for m in alice["messages"] if m.type == "human"))
        asked.append(sum(1 for m in bob["messages"] if m.type == "human"))
        bob_saw_alice = any("alice" in str(m.content) for m in bob["messages"])

    assert asked == [2, 1]  # Alice's two questions on her thread, Bob's one on his
    assert not bob_saw_alice


# --- failures mid-stream (2.1.8) -----------------------------------------------------

import logging

from langchain_core.messages import ToolMessage

from finai.api import streaming


class StrictPlanner:
    """Refuses a conversation with an unanswered tool request, as the Claude
    API does (400 "tool_use ids were found without tool_result blocks")."""

    def __init__(self) -> None:
        self.questions = 0

    def __call__(self, messages, config=None) -> AIMessage:
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        for m in messages:
            for call in getattr(m, "tool_calls", None) or []:
                if call["id"] not in answered:
                    if m is messages[-1]:
                        break  # the request just made; the tools node answers it
                    raise ValueError(
                        "tool_use ids were found without tool_result blocks"
                    )
        if messages[-1].type == "human":
            self.questions += 1
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "id": f"c{self.questions}",
                        "name": "get_live_market_data",
                        "args": {"ticker": "NVDA"},
                    }
                ],
            )
        return AIMessage(RIGHT)


def _strict_client(planner: StrictPlanner) -> TestClient:
    @asynccontextmanager
    async def graph():
        yield build_graph(
            RunnableLambda(planner), checkpointer=InMemorySaver(), critic=None
        )

    return TestClient(create_app(graph), headers=AUTH)


def test_a_failure_mid_stream_ends_with_an_error_event(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(ticker):
        raise RuntimeError("internal detail: qdrant at 10.0.0.5 refused")

    monkeypatch.setattr(market_data, "get_live_market_data", broken)
    with caplog.at_level(logging.ERROR, logger="finai.api.streaming"):
        events = _ask(_strict_client(StrictPlanner()), session_id="s1")

    kind, data = events[-1]
    assert kind == "error"
    assert data["message"] == streaming.ERROR_MESSAGE
    assert data["session_id"] == "s1"
    assert "done" not in [k for k, _ in events]
    assert "10.0.0.5" not in json.dumps(events)  # nothing internal reaches the client
    # The same ID is in the log, with the traceback.
    assert data["error_id"] in caplog.text
    assert "RuntimeError" in caplog.text


def test_the_thread_still_works_after_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the repair, the next question would hit the dangling tool call."""
    planner = StrictPlanner()
    client = _strict_client(planner)
    real_quote = market_data.get_live_market_data
    calls = {"n": 0}

    def flaky(ticker):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("timeout")
        return real_quote(ticker)

    monkeypatch.setattr(market_data, "get_live_market_data", flaky)
    with client:
        first = _events(
            client.post("/api/v1/chat", json={"message": "q1", "session_id": "s"}).text
        )
        second = _events(
            client.post("/api/v1/chat", json={"message": "q2", "session_id": "s"}).text
        )
        state = client.app.state.graph.get_state(
            {"configurable": {"thread_id": "demo:s"}}
        )

    assert first[-1][0] == "error"
    assert second[-1][0] == "done"
    assert "".join(d["text"] for k, d in second if k == "token") == RIGHT
    # The failed question was closed off: an error result and a short reply.
    results = [m for m in state.values["messages"] if isinstance(m, ToolMessage)]
    assert results[0].content == streaming.FAILED_RESULT
    assert results[0].status == "error"
    assert any(m.content == streaming.FAILED_REPLY for m in state.values["messages"])
    assert state.next == ()  # nothing left pending


def test_a_model_failure_before_any_tool_also_reports_and_recovers() -> None:
    attempts = {"n": 0}
    good = StrictPlanner()

    def planner(messages, config=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("model timed out")
        return good(messages, config)

    @asynccontextmanager
    async def graph():
        yield build_graph(
            RunnableLambda(planner), checkpointer=InMemorySaver(), critic=None
        )

    with TestClient(create_app(graph), headers=AUTH) as client:
        first = _events(
            client.post("/api/v1/chat", json={"message": "q1", "session_id": "s"}).text
        )
        second = _events(
            client.post("/api/v1/chat", json={"message": "q2", "session_id": "s"}).text
        )

    assert first[-1][0] == "error"
    assert second[-1][0] == "done"


def test_a_failed_repair_still_sends_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(ticker):
        raise RuntimeError("down")

    async def repair_fails(graph, config):
        raise ConnectionError("redis down too")

    monkeypatch.setattr(market_data, "get_live_market_data", broken)
    monkeypatch.setattr(streaming, "repair_thread", repair_fails)
    events = _ask(_strict_client(StrictPlanner()))
    assert events[-1][0] == "error"


# --- API-key authentication (2.1.9) ---------------------------------------------------


def _without_key() -> TestClient:
    client = _client()
    del client.headers["X-API-Key"]
    return client


def test_a_missing_key_is_401_with_a_challenge() -> None:
    with _without_key() as client:
        no_header = client.post("/api/v1/chat", json={"message": "q"})
        empty = client.post(
            "/api/v1/chat", json={"message": "q"}, headers={"X-API-Key": ""}
        )
    for r in (no_header, empty):
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "APIKey"


def test_a_wrong_key_is_401() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": "q"},
            headers={"X-API-Key": "nope-0123456789abcdef"},
        )
    assert (response.status_code, response.json()) == (
        401,
        {"detail": "Invalid API key"},
    )


def test_the_key_decides_the_user() -> None:
    """Bob's key makes Bob the user: his thread, whatever the session ID."""
    with _client() as client:
        client.post(
            "/api/v1/chat",
            json={"message": "q", "session_id": "s"},
            headers={"X-API-Key": "bob-key-0123456789abcd"},
        )
        graph = client.app.state.graph
        bob = graph.get_state({"configurable": {"thread_id": "bob:s"}}).values
        demo = graph.get_state({"configurable": {"thread_id": "demo:s"}}).values
    assert bob.get("user_id") == "bob"
    assert not demo  # nothing was stored under the demo user


def test_with_no_keys_configured_every_request_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finai.config import settings

    monkeypatch.setattr(settings, "API_KEYS", None)
    with _client() as client:
        response = client.post("/api/v1/chat", json={"message": "q"})
    assert response.status_code == 401


def test_probes_still_need_no_key() -> None:
    with _without_key() as client:
        assert client.get("/healthz").status_code == 200
