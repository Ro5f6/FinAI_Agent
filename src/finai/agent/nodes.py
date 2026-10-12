"""Graph nodes.

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

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig

from finai.agent.prompts import SYSTEM_PROMPT
from finai.agent.state import AgentState

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
