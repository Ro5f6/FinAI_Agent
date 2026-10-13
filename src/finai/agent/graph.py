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

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from finai.agent.nodes import (
    Planner,
    make_planner_node,
    make_tool_node,
    stop_at_limit,
)
from finai.agent.state import AgentState
from finai.config import settings

PLANNER = "planner"
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


def build_graph(
    model: Planner | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the agent (1.3.12).

    ``model`` defaults to the tool-bound planner from settings (1.3.7); tests
    pass a scripted stand-in. ``checkpointer`` keeps threads between questions
    (1.3.14); without one each run starts empty.
    """
    if model is None:
        from finai.agent.model import make_planner

        model = make_planner()

    graph = StateGraph(AgentState)
    graph.add_node(PLANNER, make_planner_node(model))
    graph.add_node(TOOLS, make_tool_node())
    graph.add_node(STOP_AT_LIMIT, stop_at_limit)

    graph.add_edge(START, PLANNER)
    graph.add_conditional_edges(
        PLANNER, route_after_planner, [TOOLS, STOP_AT_LIMIT, END]
    )
    graph.add_edge(TOOLS, PLANNER)
    graph.add_edge(STOP_AT_LIMIT, END)
    return graph.compile(checkpointer=checkpointer)
