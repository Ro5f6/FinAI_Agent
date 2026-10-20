"""Tests for routing after the planner (1.3.11) and the step-limit stop."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from finai.agent.graph import STOP_AT_LIMIT, TOOLS, route_after_planner
from finai.agent.nodes import STEP_LIMIT_RESULT, stop_at_limit
from finai.config import settings


def _asks_for(*tickers: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"id": f"t{i}", "name": "get_live_market_data", "args": {"ticker": t}}
            for i, t in enumerate(tickers)
        ],
    )


def _state(last, loop_count: int) -> dict:
    return {
        "messages": [HumanMessage("q"), last],
        "loop_count": loop_count,
        "retrieved_docs": [],
        "user_id": "demo",
    }


def test_answer_ends_the_run() -> None:
    assert route_after_planner(_state(AIMessage("NVDA is at $226.30."), 1)) == END


@pytest.mark.parametrize("loop_count", [1, settings.MAX_AGENT_LOOPS - 1])
def test_tool_request_under_the_limit_goes_to_tools(loop_count: int) -> None:
    assert route_after_planner(_state(_asks_for("NVDA"), loop_count)) == TOOLS


@pytest.mark.parametrize(
    "loop_count", [settings.MAX_AGENT_LOOPS, settings.MAX_AGENT_LOOPS + 3]
)
def test_tool_request_at_the_limit_stops(loop_count: int) -> None:
    assert route_after_planner(_state(_asks_for("NVDA"), loop_count)) == STOP_AT_LIMIT


def test_limit_follows_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MAX_AGENT_LOOPS", 2)

    assert route_after_planner(_state(_asks_for("NVDA"), 1)) == TOOLS
    assert route_after_planner(_state(_asks_for("NVDA"), 2)) == STOP_AT_LIMIT


def test_stop_answers_every_pending_call_and_the_user() -> None:
    update = stop_at_limit(_state(_asks_for("NVDA", "AAPL"), settings.MAX_AGENT_LOOPS))
    *results, answer = update["messages"]

    assert [r.tool_call_id for r in results] == ["t0", "t1"]
    assert all(isinstance(r, ToolMessage) and r.status == "error" for r in results)
    assert all(r.content == STEP_LIMIT_RESULT for r in results)
    assert isinstance(answer, AIMessage)
    assert not answer.tool_calls
    assert str(settings.MAX_AGENT_LOOPS) in answer.content


def test_thread_stays_valid_for_the_next_question() -> None:
    """Every tool_use in the next request is followed by its tool_result,
    which the Claude API requires (checked offline on the real request)."""
    from finai.agent.model import make_planner

    pending = AIMessage(
        content=[
            {
                "type": "tool_use",
                "id": "t0",
                "name": "get_live_market_data",
                "input": {"ticker": "NVDA"},
            }
        ],
        tool_calls=[
            {"id": "t0", "name": "get_live_market_data", "args": {"ticker": "NVDA"}}
        ],
    )
    state = _state(pending, settings.MAX_AGENT_LOOPS)
    history = [
        *state["messages"],
        *stop_at_limit(state)["messages"],
        HumanMessage("next question"),
    ]

    planner = make_planner()
    payload = planner.bound._get_request_payload(history, **planner.kwargs)

    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    result_block = payload["messages"][2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "t0"


# --- the compiled graph (1.3.12), end to end with a scripted model ----------

import datetime as dt
import json

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from finai.agent.graph import build_graph
from finai.agent.nodes import STEP_LIMIT_ANSWER
from finai.agent.state import turn_input
from finai.tools import filings, market_data
from finai.tools.filings import TenKPassage, TenKSearchResult
from finai.tools.market_data import MarketData


class _Script:
    """A stand-in planner: returns scripted replies in order and records what
    it was shown, so a test can check what reached the model."""

    def __init__(self, *replies: AIMessage) -> None:
        self.replies = list(replies)
        self.seen: list[list] = []

    def __call__(self, messages, config=None) -> AIMessage:
        self.seen.append(list(messages))
        return self.replies.pop(0) if self.replies else AIMessage("Done.")

    def runnable(self) -> RunnableLambda:
        return RunnableLambda(self)


def _call(name: str, call_id: str, **args) -> dict:
    return {"id": call_id, "name": name, "args": args}


@pytest.fixture
def stub_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    quote = MarketData(
        ticker="NVDA",
        name="NVIDIA Corp",
        price=226.3,
        price_currency="USD",
        change_pct=-1.2997,
        market_cap=5.5e12,
        market_cap_currency="USD",
        week_52_high=243.37,
        week_52_low=164.27,
        pe_ttm=28.61,
        pe_annual=46.2,
        pe_source="price / EPS",
        eps_ttm=7.9108,
        eps_ttm_period_end=dt.date(2026, 7, 26),
        eps_annual=4.8979,
        eps_annual_period_end=dt.date(2026, 1, 25),
        eps_currency="USD",
        eps_per="this ticker's share",
        quote_time=dt.datetime(2026, 10, 12, 16, 54, 56, tzinfo=dt.UTC),
    )
    monkeypatch.setattr(market_data, "get_live_market_data", lambda ticker: quote)

    def search(ticker, question, section, fiscal_year, k, *, user_id=None):
        passage = TenKPassage(
            text="The possibility of additional export controls has negatively impacted demand.",
            ticker=ticker,
            form="10-K",
            fiscal_year=2026,
            filing_date="2026-02-25",
            section="Item 1A",
            accession="0001045810-26-000021",
            chunk_index=15,
            score=0.69,
            relevance=0.78,
        )
        return TenKSearchResult(
            status="ok",
            ticker=ticker,
            fiscal_year=2026,
            section=section,
            available_years=[2026, 2025, 2024],
            passages=[passage],
        )

    monkeypatch.setattr(filings, "query_company_10k", search)


def _ask(graph, question: str, config: dict | None = None) -> dict:
    config = config or {"configurable": {"user_id": "demo", "thread_id": "t1"}}
    return graph.invoke({**turn_input(question, "demo"), "retrieved_docs": []}, config)


def test_one_tool_then_an_answer(stub_tools) -> None:
    script = _Script(
        AIMessage("", tool_calls=[_call("get_live_market_data", "c1", ticker="NVDA")]),
        AIMessage(
            "NVIDIA's trailing P/E is 28.61 [Finnhub quote, 2026-10-12 16:54 UTC]."
        ),
    )

    out = _ask(build_graph(script.runnable()), "What's NVDA's P/E?")

    kinds = [type(m).__name__ for m in out["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert out["messages"][-1].content.startswith("NVIDIA's trailing P/E is 28.61")
    assert out["loop_count"] == 2
    # The second planner call saw the tool result.
    assert json.loads(script.seen[1][-1].content)["pe_ttm"] == 28.61


def test_parallel_tools_and_passages_for_the_verifier(stub_tools) -> None:
    script = _Script(
        AIMessage(
            "",
            tool_calls=[
                _call(
                    "query_company_10k", "c1", ticker="NVDA", question="export controls"
                ),
                _call("get_live_market_data", "c2", ticker="NVDA"),
                _call("get_user_portfolio", "c3"),
            ],
        ),
        AIMessage("Answer with citations."),
    )

    out = _ask(
        build_graph(script.runnable()), "How exposed is NVDA to export controls?"
    )

    results = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in results] == [
        "query_company_10k",
        "get_live_market_data",
        "get_user_portfolio",
    ]
    assert [d["chunk_index"] for d in out["retrieved_docs"]] == [15]
    assert json.loads(results[2].content)["holdings"][1]["ticker"] == "NVDA"


def test_a_model_that_never_stops_is_stopped_at_the_limit(stub_tools) -> None:
    # A fresh message per reply: add_messages treats a reused message ID as an
    # update to that message, not a new one, as a real model's never would be.
    script = _Script(
        *[
            AIMessage(
                "", tool_calls=[_call("get_live_market_data", f"c{i}", ticker="NVDA")]
            )
            for i in range(20)
        ]
    )

    out = _ask(build_graph(script.runnable()), "Tell me everything.")

    assert out["loop_count"] == settings.MAX_AGENT_LOOPS
    assert len(script.seen) == settings.MAX_AGENT_LOOPS
    assert out["messages"][-1].content == STEP_LIMIT_ANSWER.format(
        limit=settings.MAX_AGENT_LOOPS
    )
    # Every tool call in the thread has a result.
    called = {
        c["id"]
        for m in out["messages"]
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    }
    answered = {m.tool_call_id for m in out["messages"] if isinstance(m, ToolMessage)}
    assert called <= answered


def test_a_second_question_on_the_same_thread(stub_tools) -> None:
    """With a checkpointer the thread continues, and the step count restarts."""
    script = _Script(
        AIMessage("", tool_calls=[_call("get_live_market_data", "c1", ticker="NVDA")]),
        AIMessage("First answer."),
        AIMessage("Second answer."),
    )
    graph = build_graph(script.runnable(), checkpointer=InMemorySaver())

    _ask(graph, "What's NVDA's P/E?")
    out = _ask(graph, "And what was the quote time?")

    assert out["messages"][-1].content == "Second answer."
    assert out["loop_count"] == 1
    # The second question's planner call saw the whole thread.
    assert [type(m).__name__ for m in script.seen[-1]] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
        "HumanMessage",
    ]


def test_build_graph_defaults_to_the_real_planner() -> None:
    """Builds without calling the API: construction is offline."""
    graph = build_graph()

    assert {"planner", "tools", "stop_at_limit"} <= set(graph.get_graph().nodes)


def test_saved_diagram_matches_the_graph() -> None:
    """docs/agent-graph.md is regenerated with `python -m finai.agent.graph`."""
    import pathlib
    import re

    from finai.agent.graph import mermaid

    doc = pathlib.Path(__file__).parent.parent / "docs" / "agent-graph.md"
    saved = re.search(r"```mermaid\n(.*?)\n```", doc.read_text(), re.DOTALL)

    assert saved is not None
    assert saved.group(1) == mermaid().rstrip("\n")
