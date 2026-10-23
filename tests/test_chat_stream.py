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

    return TestClient(create_app(graph))


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
