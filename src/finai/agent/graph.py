"""The agent graph: plan, run tools, repeat, within a step limit.

After each planner step (1.3.11):

    reply asks for tools, under the limit  ->  tools, then back to the planner
    reply asks for tools, limit reached    ->  stop_at_limit, then end
    reply asks for nothing                 ->  end (the reply is the answer)

The middle route is an addition to the plan, which ended the run at the limit:
that left tool calls without results in the saved thread, and the Claude API
rejects such a conversation, so the thread's next question would fail.
"""

from typing import Literal

from langgraph.graph import END

from finai.agent.state import AgentState
from finai.config import settings

TOOLS = "tools"
STOP_AT_LIMIT = "stop_at_limit"

Route = Literal["tools", "stop_at_limit", "__end__"]


def route_after_planner(state: AgentState) -> Route:
    """Where the run goes after the planner's latest reply."""
    if not getattr(state["messages"][-1], "tool_calls", None):
        return END
    if state["loop_count"] < settings.MAX_AGENT_LOOPS:
        return TOOLS
    return STOP_AT_LIMIT
