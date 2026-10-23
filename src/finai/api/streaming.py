"""Turning an agent run into the events of the answer stream (2.1.6).

The stream shows progress while the agent works and then the answer, once
it has been verified (D-053). Claude's drafts are never sent: the verifier
may reject one and send it back, and a reader shouldn't see a figure that
is then withdrawn. So the graph runs with ``stream_mode="updates"``, which
yields each node's output as it finishes, and each update becomes events:

    planner asks for tools     ->  tool_start, one per call
    tools return               ->  tool_end, one per result, with a summary
    planner writes a draft     ->  status "verifying" (the text is not sent)
    verifier sends it back     ->  status "revising"
    run ends                   ->  the verified answer as token pieces, done

Verified against langgraph 1.2.14: each update is ``{node: update}``, the
planner's draft is an ``AIMessage`` without tool calls, a revise adds a
feedback message, and the final answer is read from the saved state.
"""

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from finai.agent.session import answer_text, thread_config
from finai.agent.state import turn_input
from finai.agent.verification.verifier import is_feedback
from finai.api.schemas import StreamEvent

# The answer goes out in pieces of about this many characters, split at
# whitespace, so a client can render it as it arrives.
PIECE_CHARS = 48


def _summary(tool: str, output: Any) -> str:
    """One line on what a tool returned, for a progress display."""
    if not isinstance(output, dict):
        return "error"
    if output.get("status") not in (None, "ok"):
        return str(output.get("status"))
    ticker = output.get("ticker", "")
    if tool == "query_company_10k":
        section = f", {output['section']}" if output.get("section") else ""
        count = len(output.get("passages", []))
        return f"{ticker} 10-K FY{output.get('fiscal_year')}{section}: {count} passages"
    if tool == "get_live_market_data":
        return f"{ticker} at {output.get('price')} {output.get('price_currency', '')}".strip()
    if tool == "get_financial_history":
        years = output.get("fiscal_years") or []
        return f"{ticker} fiscal {min(years)}-{max(years)}" if years else ticker
    if tool == "get_price_history":
        return f"{ticker} {output.get('period')}: {output.get('start_date')} to {output.get('end_date')}"
    if tool == "get_portfolio_value":
        return f"{len(output.get('positions', []))} holdings valued"
    if tool == "get_user_portfolio":
        return f"{len(output.get('holdings', []))} holdings"
    return "done"


def pieces(text: str, size: int = PIECE_CHARS) -> list[str]:
    """``text`` split at whitespace into pieces of about ``size`` characters."""
    out: list[str] = []
    current = ""
    for word in re.findall(r"\S+\s*|\s+", text):
        if current and len(current) + len(word) > size:
            out.append(current)
            current = ""
        current += word
    if current:
        out.append(current)
    return out


def _events_for(node: str, update: dict[str, Any]) -> list[StreamEvent]:
    events = []
    for message in update.get("messages", []):
        if node == "planner" and isinstance(message, AIMessage):
            if message.tool_calls:
                events += [
                    StreamEvent(
                        type="tool_start",
                        data={"tool": c["name"], "args": c["args"], "call_id": c["id"]},
                    )
                    for c in message.tool_calls
                ]
            else:
                events.append(StreamEvent(type="status", data={"stage": "verifying"}))
        elif node == "tools" and isinstance(message, ToolMessage):
            try:
                output = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                output = message.content
            events.append(
                StreamEvent(
                    type="tool_end",
                    data={
                        "tool": message.name,
                        "call_id": message.tool_call_id,
                        "status": message.status,
                        "summary": _summary(message.name, output),
                    },
                )
            )
        elif (
            node == "verifier"
            and isinstance(message, HumanMessage)
            and is_feedback(message)
        ):
            events.append(StreamEvent(type="status", data={"stage": "revising"}))
    return events


async def answer_events(
    graph: CompiledStateGraph,
    question: str,
    *,
    user_id: str,
    thread_id: str,
    session_id: str,
) -> AsyncIterator[StreamEvent]:
    """The events of one answer, as the agent produces them."""
    config = thread_config(user_id=user_id, thread_id=thread_id)
    async for chunk in graph.astream(
        turn_input(question, user_id), config, stream_mode="updates"
    ):
        for node, update in chunk.items():
            for event in _events_for(node, update or {}):
                yield event
    state = (await graph.aget_state(config)).values
    for piece in pieces(answer_text(state)):
        yield StreamEvent(type="token", data={"text": piece})
    verdict = state.get("verdict") or {}
    yield StreamEvent(
        type="done",
        data={
            "session_id": session_id,
            "verdict": verdict.get("status"),
            "revisions": state.get("revision_count", 0),
        },
    )
