"""Graph nodes.

The tool node is LangGraph's ``ToolNode`` plus one step. Verified against
langgraph-prebuilt 1.1.0 (docs/CONTEXT.md), ``ToolNode`` already runs a
reply's tool calls concurrently, serialises dict results to JSON, injects the
run's config (the portfolio tool's user), returns invalid arguments and
unknown tool names to the model as error messages so it can correct itself,
and raises on any other exception so a bug isn't turned into model input. The
added step records what the tools returned for the verifier (1.5): every
result, parsed, with the arguments it was called with, in ``tool_results``;
and the passages of each successful 10-K search in ``retrieved_docs``, also
used by the evals (6.2).

The planner reads the conversation and either answers or asks for tools. The
system prompt is prepended on every call rather than kept in the state: the
state is checkpointed per thread (1.3.14), so a stored prompt would be saved
again with every turn and could go stale when the prompt changes. As a
``SystemMessage`` it goes to the API's top-level ``system`` field.

The model's reply goes into the state unchanged. On Sonnet 5.5 it carries
thinking blocks that must be sent back as they came, signature included, on
the next call of the same tool loop; ``ChatAnthropic`` does that when the
``AIMessage`` is passed back (verified offline, docs/CONTEXT.md).
"""

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from finai.agent.prompts import SYSTEM_PROMPT
from finai.agent.state import AgentState
from finai.agent.verification.verifier import (
    Critic,
    caveat,
    feedback_message,
    verify,
)
from finai.config import settings
from finai.tools.agent_tools import TOOLS

Planner = Runnable[LanguageModelInput, BaseMessage]


def make_planner_node(
    model: Planner,
) -> Callable[[AgentState, RunnableConfig], dict[str, Any]]:
    """The planner node for ``model``: the tool-bound chat model from 1.3.7,
    or a stand-in in tests."""

    def planner(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Ask the model for its next step and count the step."""
        reply = model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]], config
        )
        return {"messages": [reply], "loop_count": state["loop_count"] + 1}

    return planner


def _passages(message: ToolMessage) -> list[dict[str, Any]]:
    """The passages in a successful ``query_company_10k`` result, else []."""
    if message.name != "query_company_10k" or message.status != "success":
        return []
    return json.loads(message.content).get("passages", [])


def _tool_result(message: ToolMessage, args: dict[str, Any]) -> dict[str, Any]:
    """One tool's output as the verifier reads it.

    Tools return dicts, which ``ToolNode`` serialises to JSON; an error
    (invalid arguments, an unknown tool) is plain text, kept as it is.
    """
    try:
        output: Any = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        output = message.content
    return {
        "tool": message.name,
        "tool_call_id": message.tool_call_id,
        "args": args,
        "status": message.status,
        "output": output,
    }


def make_tool_node(
    tools: Sequence[BaseTool] = TOOLS,
) -> Callable[[AgentState, RunnableConfig], dict[str, Any]]:
    """The tool node: run the requested tools, keep their outputs and passages."""
    tool_node = ToolNode(tools)

    def run_tools(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        results = tool_node.invoke(state, config)["messages"]
        calls = state["messages"][-1].tool_calls
        args = {call["id"]: call["args"] for call in calls}
        return {
            "messages": results,
            "tool_results": [
                _tool_result(m, args.get(m.tool_call_id, {})) for m in results
            ],
            "retrieved_docs": [p for m in results for p in _passages(m)],
        }

    return run_tools


STEP_LIMIT_RESULT = "Not run: the step limit for this question was reached."
STEP_LIMIT_ANSWER = (
    "I stopped before finishing: this question needed more lookups than I'm "
    "allowed for one answer ({limit}). Asking about one company or one aspect "
    "at a time usually works."
)


def stop_at_limit(state: AgentState) -> dict[str, Any]:
    """End a run whose last reply still asks for tools at the step limit.

    Ending there would leave the user without an answer and leave tool calls
    without results in the checkpointed thread; the Claude API rejects a
    conversation with an unanswered tool call, so the thread's next question
    would fail. Each pending call gets a result saying it wasn't run, and the
    user gets a short explanation.
    """
    last = state["messages"][-1]
    results = [
        ToolMessage(
            content=STEP_LIMIT_RESULT,
            tool_call_id=call["id"],
            name=call["name"],
            status="error",
        )
        for call in getattr(last, "tool_calls", [])
    ]
    answer = AIMessage(content=STEP_LIMIT_ANSWER.format(limit=settings.MAX_AGENT_LOOPS))
    return {"messages": [*results, answer]}


def make_verifier_node(
    critic: Critic | None,
) -> Callable[[AgentState], dict[str, Any]]:
    """The verifier node (1.5.11-1.5.13): check the draft, then decide.

    pass: the draft is the answer. revise, with revisions and planner steps
    left: the issues go back to the planner as a feedback message. revise
    with either limit reached: the draft is returned with a caveat listing
    the claims that couldn't be verified, replacing the draft (same message
    ID) so the thread keeps one final answer.
    """

    def verifier(state: AgentState) -> dict[str, Any]:
        draft = state["messages"][-1]
        verdict = verify(draft.text, state, critic)
        update: dict[str, Any] = {"verdict": verdict.model_dump()}
        if verdict.status == "pass":
            return update
        can_revise = (
            state["revision_count"] < settings.MAX_REVISIONS
            and state["loop_count"] < settings.MAX_AGENT_LOOPS
        )
        if can_revise:
            update["revision_count"] = state["revision_count"] + 1
            update["messages"] = [feedback_message(verdict.issues)]
        else:
            update["messages"] = [
                AIMessage(content=draft.text + caveat(verdict.issues), id=draft.id)
            ]
        return update

    return verifier
