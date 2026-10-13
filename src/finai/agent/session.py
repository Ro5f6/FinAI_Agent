"""Asking the agent a question on a thread.

``ask`` is the one way callers (the API, the test prompts, evals) start a
turn. It builds the input with ``turn_input``, so every question resets the
step count (a checkpointed thread would otherwise start later questions near
the limit), and puts ``thread_id`` and ``user_id`` in the config: the
checkpointer keys the saved conversation by ``thread_id``, and the portfolio
tool reads the user from the config, never from the model.

Verified against langgraph 1.2.14: a checkpointed graph invoked without a
``thread_id`` raises a bare ``KeyError: 'thread_id'``; ``ask`` refuses it
first with a clear message. ``retrieved_docs`` starts as [] on a new thread.
"""

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from finai.agent.state import turn_input


class NoAnswerError(LookupError):
    """The thread's last message isn't an answer from the agent."""


def thread_config(*, user_id: str, thread_id: str) -> dict[str, Any]:
    """The run config for one user's thread."""
    if not user_id or not thread_id:
        raise ValueError("user_id and thread_id are both required")
    return {"configurable": {"user_id": user_id, "thread_id": thread_id}}


def ask(
    graph: CompiledStateGraph, question: str, *, user_id: str, thread_id: str
) -> dict[str, Any]:
    """Run one question on ``thread_id`` and return the thread's state after it."""
    config = thread_config(user_id=user_id, thread_id=thread_id)
    return graph.invoke(turn_input(question, user_id), config)


def answer_text(state: dict[str, Any]) -> str:
    """The text of the agent's final reply in ``state``."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        raise NoAnswerError(
            f"the thread ends with a {type(last).__name__}, not an answer"
        )
    return last.text
