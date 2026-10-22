"""What the agent graph carries from step to step.

A field without a reducer is replaced by whatever a step returns; one with a
reducer is merged. Verified against langgraph 1.2.14 (docs/CONTEXT.md):
``messages`` appends through ``add_messages``, which also gives each message
an ID; ``retrieved_docs`` appends through ``operator.add``, so the tool step
returns only the passages it just retrieved.

The state is checkpointed per thread (1.3.14), so it outlives a single
question. ``loop_count``, ``revision_count`` and ``verdict`` are reset by
each new question (``turn_input``); the documents and tool results
accumulate over the thread on purpose, since evidence found earlier in a
conversation remains evidence the verifier (1.5) may check.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State of one conversation thread."""

    # The conversation: user questions, model replies, tool calls and results.
    messages: Annotated[list[AnyMessage], add_messages]
    # 10-K passages retrieved in this thread, as returned by query_company_10k,
    # kept for the verifier (1.5) and the retrieval evals (6.2).
    retrieved_docs: Annotated[list[dict[str, Any]], operator.add]
    # Every tool call's structured output in this thread (1.5.1): the
    # verifier matches the answer's numbers against these. Each is
    # {"tool", "tool_call_id", "status", "output"}; the call ID ties it to its
    # message, so the current question's results can be told apart.
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    # The verifier's last finding for the current answer: {"status": "pass" |
    # "revise", "issues": [...]}. None until the first verification.
    verdict: dict[str, Any] | None
    # Times the current answer was sent back to the planner; capped by
    # MAX_REVISIONS (1.5.12).
    revision_count: int
    # Planner steps taken for the current question; capped by MAX_AGENT_LOOPS.
    loop_count: int
    # Set from authentication by the caller, never from the model.
    user_id: str


def turn_input(question: str, user_id: str) -> dict[str, Any]:
    """The update that starts a new question on a thread.

    Resets ``loop_count``: without it, a thread's later questions would start
    near the loop cap; and ``revision_count`` and ``verdict``, which belong
    to one answer. ``retrieved_docs`` and ``tool_results`` accumulate.
    """
    return {
        "messages": [HumanMessage(content=question)],
        "loop_count": 0,
        "revision_count": 0,
        "verdict": None,
        "user_id": user_id,
    }
