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
