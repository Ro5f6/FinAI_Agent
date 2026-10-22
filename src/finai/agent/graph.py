"""The agent graph: plan, run tools, verify, within limits.

After each planner step (1.3.11, 1.5.11):

    reply asks for tools, under the limit  ->  tools, then back to the planner
    reply asks for tools, limit reached    ->  stop_at_limit, then end
    reply asks for nothing                 ->  verifier (the reply is a draft)

After the verifier (1.5.12):

    pass                                   ->  end (the draft is the answer)
    revise, revisions and steps left       ->  planner, with the issues as feedback
    revise, a limit reached                ->  end, the draft with a caveat

Termination: every revision adds one to ``revision_count`` (at most
``MAX_REVISIONS``) and every planner step one to ``loop_count`` (at most
``MAX_AGENT_LOOPS``, across revisions, 1.5.13); neither is reset within a
question, so a question ends after at most ``MAX_AGENT_LOOPS`` planner steps
and ``MAX_REVISIONS`` + 1 verifications.

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
    make_verifier_node,
    stop_at_limit,
)
from finai.agent.state import AgentState
from finai.agent.verification.verifier import Critic, is_feedback
from finai.config import settings

PLANNER = "planner"
TOOLS = "tools"
STOP_AT_LIMIT = "stop_at_limit"
VERIFIER = "verifier"

Route = Literal["tools", "stop_at_limit", "verifier"]


def route_after_planner(state: AgentState) -> Route:
    """Where the run goes after the planner's latest reply."""
    if not getattr(state["messages"][-1], "tool_calls", None):
        return VERIFIER
    if state["loop_count"] < settings.MAX_AGENT_LOOPS:
        return TOOLS
    return STOP_AT_LIMIT


def route_after_verifier(state: AgentState) -> Literal["planner", "__end__"]:
    """Back to the planner when the verifier added feedback; else the end."""
    return PLANNER if is_feedback(state["messages"][-1]) else END


def build_graph(
    model: Planner | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    critic: Critic | None | Literal["default"] = "default",
) -> CompiledStateGraph:
    """Compile the agent (1.3.12, 1.5.11).

    ``model`` defaults to the tool-bound planner from settings (1.3.7); tests
    pass a scripted stand-in. ``checkpointer`` keeps threads between questions
    (1.3.14); without one each run starts empty. ``critic`` is the verifier's
    LLM critic: by default the Haiku critic (1.5.10) with the real planner,
    and none with a scripted one, so tests make no model call unless they
    pass a critic; ``None`` runs the deterministic checks only.
    """
    if critic == "default":
        if model is None:
            from finai.agent.verification.critic import make_critic

            critic = make_critic()
        else:
            critic = None
    if model is None:
        from finai.agent.model import make_planner

        model = make_planner()

    graph = StateGraph(AgentState)
    graph.add_node(PLANNER, make_planner_node(model))
    graph.add_node(TOOLS, make_tool_node())
    graph.add_node(STOP_AT_LIMIT, stop_at_limit)
    graph.add_node(VERIFIER, make_verifier_node(critic))

    graph.add_edge(START, PLANNER)
    graph.add_conditional_edges(
        PLANNER, route_after_planner, [TOOLS, STOP_AT_LIMIT, VERIFIER]
    )
    graph.add_edge(TOOLS, PLANNER)
    graph.add_conditional_edges(VERIFIER, route_after_verifier, [PLANNER, END])
    graph.add_edge(STOP_AT_LIMIT, END)
    return graph.compile(checkpointer=checkpointer)


def mermaid() -> str:
    """The compiled graph as a Mermaid diagram, as saved in docs/agent-graph.md.

    Builds the real graph; construction makes no API call.
    """
    return build_graph().get_graph().draw_mermaid()


if __name__ == "__main__":
    print(mermaid())
