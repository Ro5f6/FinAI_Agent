"""What the agent graph carries from step to step.

A field without a reducer is replaced by whatever a step returns; one with a
reducer is merged. Verified against langgraph 1.2.14 (docs/CONTEXT.md):
``messages`` appends through ``add_messages``, which also gives each message
an ID; ``retrieved_docs`` appends through ``operator.add``, so the tool step
returns only the passages it just retrieved.

The state is checkpointed per thread (1.3.14), so it outlives a single
question. ``loop_count`` is reset by each new question (``turn_input``); the
documents accumulate over the thread on purpose, since passages found
earlier in a conversation remain evidence the verifier (1.5) may check.
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
    # Planner steps taken for the current question; capped by MAX_AGENT_LOOPS.
    loop_count: int
    # Set from authentication by the caller, never from the model.
    user_id: str


def turn_input(question: str, user_id: str) -> dict[str, Any]:
    """The update that starts a new question on a thread.

    Resets ``loop_count``: without it, a thread's later questions would start
    near the loop cap. ``retrieved_docs`` is left to accumulate.
    """
    return {
        "messages": [HumanMessage(content=question)],
        "loop_count": 0,
        "user_id": user_id,
    }
