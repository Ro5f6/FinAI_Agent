"""Tests for AgentState: how each field merges as the graph runs.

A two-step graph stands in for planner and tools, so the reducers are
exercised by LangGraph itself rather than called directly.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from finai.agent.state import AgentState, turn_input

_PASSAGE_A = {"ticker": "AAPL", "fiscal_year": 2025, "section": "Item 1A", "text": "A"}
_PASSAGE_B = {"ticker": "AAPL", "fiscal_year": 2025, "section": "Item 7", "text": "B"}


def _planner(state: AgentState) -> dict:
    return {
        "messages": [AIMessage(content=f"step {state['loop_count'] + 1}")],
        "loop_count": state["loop_count"] + 1,
    }


def _tools(state: AgentState) -> dict:
    passage = _PASSAGE_A if state["loop_count"] == 1 else _PASSAGE_B
    return {"retrieved_docs": [passage]}


def _graph(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("planner", _planner)
    graph.add_node("tools", _tools)
    graph.add_node("planner_again", _planner)
    graph.add_node("tools_again", _tools)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "tools")
    graph.add_edge("tools", "planner_again")
    graph.add_edge("planner_again", "tools_again")
    graph.add_edge("tools_again", END)
    return graph.compile(checkpointer=checkpointer)


def test_messages_append() -> None:
    out = _graph().invoke({**turn_input("Supply risks?", "demo"), "retrieved_docs": []})

    assert [m.content for m in out["messages"]] == ["Supply risks?", "step 1", "step 2"]
    assert isinstance(out["messages"][0], HumanMessage)
    assert all(m.id for m in out["messages"])


def test_retrieved_docs_accumulate_instead_of_being_replaced() -> None:
    """Without the operator.add reducer the second tool step would leave [B]."""
    out = _graph().invoke({**turn_input("Supply risks?", "demo"), "retrieved_docs": []})

    assert out["retrieved_docs"] == [_PASSAGE_A, _PASSAGE_B]


def test_loop_count_is_replaced_by_each_step() -> None:
    out = _graph().invoke({**turn_input("Supply risks?", "demo"), "retrieved_docs": []})

    assert out["loop_count"] == 2


def test_a_new_question_resets_the_loop_count_but_keeps_the_thread() -> None:
    """With a checkpointer the state outlives the question; turn_input must
    restart the count, or later questions would begin near the cap."""
    graph = _graph(InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    graph.invoke({**turn_input("First?", "demo"), "retrieved_docs": []}, config)
    out = graph.invoke(turn_input("Second?", "demo"), config)

    assert out["loop_count"] == 2  # 0 again at the start, then two steps
    assert [m.content for m in out["messages"]][:4] == [
        "First?",
        "step 1",
        "step 2",
        "Second?",
    ]
    assert len(out["retrieved_docs"]) == 4  # both questions' passages kept


def test_turn_input_carries_the_user_from_the_caller() -> None:
    update = turn_input("Hi", "user-42")

    assert update["user_id"] == "user-42"
    assert update["loop_count"] == 0
    assert "retrieved_docs" not in update  # left to accumulate
