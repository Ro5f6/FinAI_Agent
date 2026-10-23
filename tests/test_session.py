"""Tests for asking the agent on a thread (1.3.14), with an in-memory
checkpointer and a scripted model."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from finai.agent.graph import build_graph
from finai.agent.session import NoAnswerError, answer_text, ask, thread_config
from finai.config import settings


class _Model:
    """Asks for the portfolio ``tool_steps`` times per question, then answers
    with the number of human messages it can see, so memory is visible."""

    def __init__(self, tool_steps: int = 0) -> None:
        self.tool_steps = tool_steps
        self.calls = 0
        self.seen: list[list] = []

    def __call__(self, messages, config=None) -> AIMessage:
        self.calls += 1
        self.seen.append(list(messages))
        since_question = 0
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                break
            since_question += isinstance(m, AIMessage)
        if since_question < self.tool_steps:
            return AIMessage(
                "",
                tool_calls=[
                    {"id": f"c{self.calls}", "name": "get_user_portfolio", "args": {}}
                ],
            )
        questions = sum(isinstance(m, HumanMessage) for m in messages)
        return AIMessage(f"I can see {questions} question(s).")


def _graph(model: _Model):
    return build_graph(RunnableLambda(model), checkpointer=InMemorySaver())


def test_a_thread_remembers_earlier_questions() -> None:
    graph = _graph(_Model())

    ask(graph, "What's NVDA's P/E?", user_id="demo", thread_id="t1")
    state = ask(graph, "And AMD's?", user_id="demo", thread_id="t1")

    assert answer_text(state) == "I can see 2 question(s)."
    assert [m.content for m in state["messages"] if isinstance(m, HumanMessage)] == [
        "What's NVDA's P/E?",
        "And AMD's?",
    ]


def test_threads_are_separate() -> None:
    graph = _graph(_Model())

    ask(graph, "First thread question", user_id="demo", thread_id="t1")
    state = ask(graph, "Other thread question", user_id="demo", thread_id="t2")

    assert answer_text(state) == "I can see 1 question(s)."


def test_every_question_gets_the_full_step_budget() -> None:
    """Without the reset in turn_input, the second question would start at 4
    and hit the limit of 6 before finishing."""
    model = _Model(tool_steps=4)
    graph = _graph(model)

    first = ask(graph, "Q1", user_id="demo", thread_id="t1")
    second = ask(graph, "Q2", user_id="demo", thread_id="t1")

    assert first["loop_count"] == second["loop_count"] == 5  # 4 tool steps + the answer
    assert answer_text(second) == "I can see 2 question(s)."
    assert settings.MAX_AGENT_LOOPS == 6


def test_the_user_reaches_the_tools_through_the_config() -> None:
    graph = _graph(_Model(tool_steps=1))

    state = ask(graph, "What do I hold?", user_id="demo", thread_id="t1")
    other = ask(graph, "What do I hold?", user_id="someone-else", thread_id="t2")

    def holdings(s):
        result = next(m for m in s["messages"] if isinstance(m, ToolMessage))
        return [h["ticker"] for h in json.loads(result.content)["holdings"]]

    assert holdings(state) == ["AAPL", "NVDA"]
    assert holdings(other) == []


def test_the_saved_thread_can_be_inspected() -> None:
    graph = _graph(_Model())
    ask(graph, "Q1", user_id="demo", thread_id="t1")

    snapshot = graph.get_state(thread_config(user_id="demo", thread_id="t1"))

    assert snapshot.next == ()  # finished
    assert len(snapshot.values["messages"]) == 2


@pytest.mark.parametrize(("user_id", "thread_id"), [("demo", ""), ("", "t1")])
def test_user_and_thread_are_required(user_id: str, thread_id: str) -> None:
    graph = _graph(_Model())

    with pytest.raises(ValueError, match="required"):
        ask(graph, "Q", user_id=user_id, thread_id=thread_id)


def test_answer_text_skips_reasoning_blocks() -> None:
    reply = AIMessage(
        content=[
            {"type": "thinking", "thinking": "", "signature": "s"},
            {"type": "text", "text": "NVIDIA's P/E is 28.42."},
        ]
    )

    assert (
        answer_text({"messages": [HumanMessage("q"), reply]})
        == "NVIDIA's P/E is 28.42."
    )


def test_answer_text_refuses_a_thread_without_an_answer() -> None:
    with pytest.raises(NoAnswerError, match="ToolMessage"):
        answer_text({"messages": [ToolMessage(content="{}", tool_call_id="c1")]})


# --- session to thread (2.1.7) -------------------------------------------------------

from finai.agent.session import thread_id_for


def test_the_thread_is_the_user_then_the_session() -> None:
    assert thread_id_for("alice", "3f2b9c") == "alice:3f2b9c"


def test_two_users_with_one_session_id_get_two_threads() -> None:
    assert thread_id_for("alice", "shared") != thread_id_for("bob", "shared")


def test_the_mapping_cannot_collide() -> None:
    """With no ":" in a session ID, a thread splits back into one pair."""
    pairs = [("a", "b-c"), ("a:b", "c"), ("a-b", "c"), ("x.y", "z_1")]
    threads = [thread_id_for(user, session) for user, session in pairs]
    assert len(set(threads)) == len(pairs)
    assert [tuple(t.rsplit(":", 1)) for t in threads] == pairs


@pytest.mark.parametrize(
    ("user_id", "session_id"), [("", "s"), ("u", ""), ("u", "a:b")]
)
def test_invalid_parts_are_refused(user_id: str, session_id: str) -> None:
    with pytest.raises(ValueError):
        thread_id_for(user_id, session_id)
