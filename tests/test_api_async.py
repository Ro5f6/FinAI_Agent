"""The API through ``httpx.AsyncClient``, as an async client calls it (2.1.11).

``ASGITransport`` serves the app in process but doesn't run its startup and
shutdown, so each test enters ``app.router.lifespan_context`` itself, which
builds the graph as uvicorn would. The graph is the real one with a
scripted planner and a stubbed quote: no model, no network. Tests run on
anyio's pytest plugin (installed with Starlette), on asyncio.
"""

import asyncio
import datetime as dt
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from finai.agent.graph import build_graph
from finai.api.main import create_app
from finai.api.routers import health
from finai.config import settings
from finai.tools import market_data

pytestmark = pytest.mark.anyio

KEY = "async-test-key-0123456789"
ANSWER = "NVDA is at $212.21. [Finnhub quote, 2026-10-22 20:00 UTC]"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "API_KEYS", SecretStr(f"demo:{KEY}"))
    quote = market_data.MarketData(
        ticker="NVDA", name="NVIDIA", price=212.21, price_currency="USD", change_pct=-4.4,
        market_cap=None, market_cap_currency=None, week_52_high=None, week_52_low=None,
        pe_ttm=None, pe_annual=None, pe_source=None, eps_ttm=None, eps_ttm_period_end=None,
        eps_annual=None, eps_annual_period_end=None, eps_currency=None, eps_per=None,
        quote_time=dt.datetime(2026, 10, 22, 20, 0, tzinfo=dt.UTC),
    )  # fmt: skip
    monkeypatch.setattr(market_data, "get_live_market_data", lambda ticker: quote)


def _planner(messages, config=None) -> AIMessage:
    if messages[-1].type == "human":
        return AIMessage(
            "",
            tool_calls=[
                {
                    "id": f"c{len(messages)}",
                    "name": "get_live_market_data",
                    "args": {"ticker": "NVDA"},
                }
            ],
        )
    return AIMessage(ANSWER)


@asynccontextmanager
async def _client(
    headers: dict[str, str] | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    @asynccontextmanager
    async def graph():
        yield build_graph(
            RunnableLambda(_planner), checkpointer=InMemorySaver(), critic=None
        )

    app = create_app(graph)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", headers=headers or {}
        ) as client:
            yield client


async def _stream(client: httpx.AsyncClient, body: dict) -> list[tuple[str, dict]]:
    """Read the answer's events line by line, as a streaming client does."""
    events: list[tuple[str, dict]] = []
    async with client.stream("POST", "/api/v1/chat", json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        kind = None
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                kind = line.removeprefix("event: ")
            elif line.startswith("data: ") and kind:
                events.append((kind, json.loads(line.removeprefix("data: "))))
    return events


async def test_health_and_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health, "CHECKS", {"qdrant": lambda: None, "redis": lambda: None}
    )
    async with _client() as client:
        alive = await client.get("/healthz")
        ready = await client.get("/readyz")
    assert (alive.status_code, alive.json()) == (200, {"status": "ok"})
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


async def test_a_missing_or_wrong_key_is_refused() -> None:
    async with _client() as client:
        missing = await client.post("/api/v1/chat", json={"message": "q"})
        wrong = await client.post(
            "/api/v1/chat",
            json={"message": "q"},
            headers={"X-API-Key": "wrong-0123456789abc"},
        )
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "APIKey"
    assert (wrong.status_code, wrong.json()) == (401, {"detail": "Invalid API key"})


async def test_a_streamed_answer_ends_in_done() -> None:
    async with _client({"X-API-Key": KEY}) as client:
        events = await _stream(
            client, {"message": "What's NVDA at?", "session_id": "s1"}
        )

    kinds = [kind for kind, _ in events]
    assert kinds[:3] == ["tool_start", "tool_end", "status"]
    assert "".join(data["text"] for kind, data in events if kind == "token") == ANSWER
    assert events[-1] == (
        "done",
        {"session_id": "s1", "verdict": "pass", "revisions": 0},
    )


async def test_two_questions_at_once_are_answered_separately() -> None:
    """The async server serves concurrent requests, each on its own thread."""
    async with _client({"X-API-Key": KEY}) as client:
        first, second = await asyncio.gather(
            _stream(client, {"message": "q", "session_id": "a"}),
            _stream(client, {"message": "q", "session_id": "b"}),
        )
    assert first[-1][1]["session_id"] == "a"
    assert second[-1][1]["session_id"] == "b"
    assert first[-1][0] == second[-1][0] == "done"
