"""Tests for the planner node, with a stand-in model that records its input."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from finai.agent.nodes import make_planner_node
from finai.agent.prompts import SYSTEM_PROMPT
from finai.agent.state import turn_input


class _Recorder:
    """A stand-in model: records what it receives and replies with a message."""

    def __init__(self, reply: AIMessage) -> None:
        self.reply = reply
        self.inputs: list[list] = []
        self.configs: list[RunnableConfig] = []

    def runnable(self) -> RunnableLambda:
        def call(messages, config: RunnableConfig):
            self.inputs.append(messages)
            self.configs.append(config)
            return self.reply

        return RunnableLambda(call)


def _state(**overrides) -> dict:
    return {
        **turn_input("What's NVDA's P/E?", "demo"),
        "retrieved_docs": [],
        **overrides,
    }


def test_planner_sends_the_system_prompt_then_the_conversation() -> None:
    model = _Recorder(AIMessage("NVDA trades at a P/E of 29.0."))
    state = _state()

    make_planner_node(model.runnable())(state, {})

    [sent] = model.inputs
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == SYSTEM_PROMPT
    assert sent[1:] == state["messages"]


def test_planner_returns_only_the_new_reply_and_counts_the_step() -> None:
    reply = AIMessage("Answer.")
    update = make_planner_node(_Recorder(reply).runnable())(_state(loop_count=2), {})

    assert update == {"messages": [reply], "loop_count": 3}


def test_system_prompt_is_never_stored_in_the_state() -> None:
    update = make_planner_node(_Recorder(AIMessage("Answer.")).runnable())(_state(), {})

    assert not any(isinstance(m, SystemMessage) for m in update["messages"])


def test_tool_loop_history_is_passed_back_unchanged() -> None:
    """Thinking blocks and tool calls must reach the model exactly as returned."""
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    call = AIMessage(
        content=[
            thinking,
            {
                "type": "tool_use",
                "id": "t1",
                "name": "get_live_market_data",
                "input": {"ticker": "NVDA"},
            },
        ],
        tool_calls=[
            {"id": "t1", "name": "get_live_market_data", "args": {"ticker": "NVDA"}}
        ],
    )
    result = ToolMessage(content='{"status": "ok"}', tool_call_id="t1")
    state = _state(messages=[HumanMessage("NVDA P/E?"), call, result])
    model = _Recorder(AIMessage("29.0"))

    make_planner_node(model.runnable())(state, {})

    assert model.inputs[0][2] is call
    assert model.inputs[0][2].content[0] == thinking
    assert model.inputs[0][3] is result


def test_run_config_reaches_the_model() -> None:
    """Callbacks and tracing hang off the config."""
    model = _Recorder(AIMessage("Answer."))

    make_planner_node(model.runnable())(_state(), {"tags": ["eval-run"]})

    assert "eval-run" in model.configs[0]["tags"]


def test_planner_works_with_the_real_tool_bound_model_offline() -> None:
    """The real planner builds a valid request from a node call (no API call)."""
    from finai.agent.model import make_planner

    planner = make_planner()
    captured = {}

    def capture(messages, config):
        captured["payload"] = planner.bound._get_request_payload(
            messages, **planner.kwargs
        )
        return AIMessage("ok")

    make_planner_node(RunnableLambda(capture))(_state(), {})

    payload = captured["payload"]
    assert payload["system"] == SYSTEM_PROMPT
    assert [m["role"] for m in payload["messages"]] == ["user"]
    assert len(payload["tools"]) == 3


# --- tool node (1.3.10) -------------------------------------------------------

import json

import pytest
from langgraph.graph import END, START, StateGraph

from finai.agent.nodes import make_tool_node
from finai.agent.state import AgentState
from finai.tools import filings, market_data
from finai.tools.filings import TenKPassage, TenKSearchResult


def _passage(ticker: str, year: int, index: int) -> TenKPassage:
    return TenKPassage(
        text=f"{ticker} passage {index}",
        ticker=ticker,
        form="10-K",
        fiscal_year=year,
        filing_date=f"{year}-10-31",
        section="Item 1A",
        accession=f"{ticker}-{year}",
        chunk_index=index,
        score=0.6,
        relevance=0.8,
    )


@pytest.fixture
def stub_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """query_company_10k returns two passages per indexed ticker; TSLA isn't indexed."""

    def fake(ticker, question, section, fiscal_year, k):
        if ticker == "TSLA":
            return TenKSearchResult(
                status="not_indexed",
                ticker="TSLA",
                fiscal_year=None,
                section=section,
                available_years=[],
            )
        year = fiscal_year or 2025
        return TenKSearchResult(
            status="ok",
            ticker=ticker,
            fiscal_year=year,
            section=section,
            available_years=[2025, 2024],
            passages=[_passage(ticker, year, 0), _passage(ticker, year, 1)],
        )

    monkeypatch.setattr(filings, "query_company_10k", fake)


def _run_tools(calls: list[tuple[str, dict]], config: dict | None = None) -> dict:
    graph = StateGraph(AgentState)
    graph.add_node("tools", make_tool_node())
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    request = AIMessage(
        content="",
        tool_calls=[
            {"id": f"call{i}", "name": n, "args": a} for i, (n, a) in enumerate(calls)
        ],
    )
    state = {**_state(), "messages": [HumanMessage("q"), request]}
    return graph.compile().invoke(
        state, config or {"configurable": {"user_id": "demo"}}
    )


def test_10k_passages_are_kept_for_the_verifier(stub_search) -> None:
    out = _run_tools(
        [("query_company_10k", {"ticker": "AAPL", "question": "supply risks"})]
    )

    assert [(d["ticker"], d["chunk_index"]) for d in out["retrieved_docs"]] == [
        ("AAPL", 0),
        ("AAPL", 1),
    ]
    assert out["retrieved_docs"][0]["section"] == "Item 1A"


def test_each_call_gets_its_result_message(stub_search) -> None:
    out = _run_tools(
        [
            ("query_company_10k", {"ticker": "AAPL", "question": "risks"}),
            ("get_user_portfolio", {}),
        ]
    )
    results = [m for m in out["messages"] if isinstance(m, ToolMessage)]

    assert [m.tool_call_id for m in results] == ["call0", "call1"]
    assert [m.name for m in results] == ["query_company_10k", "get_user_portfolio"]
    assert json.loads(results[1].content)["holdings"][0]["ticker"] == "AAPL"


def test_parallel_searches_keep_every_passage(stub_search) -> None:
    out = _run_tools(
        [
            ("query_company_10k", {"ticker": "AAPL", "question": "risks"}),
            (
                "query_company_10k",
                {"ticker": "MSFT", "question": "risks", "fiscal_year": 2024},
            ),
        ]
    )

    keys = {(d["ticker"], d["fiscal_year"]) for d in out["retrieved_docs"]}
    assert keys == {("AAPL", 2025), ("MSFT", 2024)}
    assert len(out["retrieved_docs"]) == 4


def test_other_tools_add_no_documents(
    stub_search, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        market_data,
        "get_live_market_data",
        lambda t: (_ for _ in ()).throw(market_data.TickerNotFoundError("no quote")),
    )

    out = _run_tools(
        [("get_live_market_data", {"ticker": "ZZZZ"}), ("get_user_portfolio", {})]
    )

    assert out["retrieved_docs"] == []


def test_unindexed_company_adds_nothing(stub_search) -> None:
    out = _run_tools([("query_company_10k", {"ticker": "TSLA", "question": "risks"})])

    assert out["retrieved_docs"] == []
    assert json.loads(out["messages"][-1].content)["status"] == "not_indexed"


def test_bad_arguments_go_back_to_the_model_as_an_error(stub_search) -> None:
    """An invalid section is the model's mistake: tell it, don't crash."""
    out = _run_tools(
        [
            (
                "query_company_10k",
                {"ticker": "AAPL", "question": "q", "section": "Risk Factors"},
            )
        ]
    )
    result = out["messages"][-1]

    assert result.status == "error"
    assert "section" in result.content
    assert out["retrieved_docs"] == []


def test_unknown_tool_goes_back_to_the_model() -> None:
    out = _run_tools([("get_stock_tips", {})])
    result = out["messages"][-1]

    assert result.status == "error"
    assert "query_company_10k" in result.content  # lists the valid tools


def test_a_bug_in_a_tool_is_not_hidden(stub_search) -> None:
    """A missing user is a server bug; it must not become a model message."""
    from finai.tools.agent_tools import MissingUserError

    with pytest.raises(MissingUserError):
        _run_tools([("get_user_portfolio", {})], {"configurable": {}})
