"""The chat endpoint: one question in, a streamed answer out (2.1.6).

``POST /api/v1/chat`` with a ``ChatRequest`` answers with server-sent
events (``finai.api.streaming``): progress while the agent works, then the
verified answer in pieces, then ``done``. The caller's identity comes from
``finai.api.deps``, never from the request body, and the conversation is
stored under that user plus the session ID (2.1.7), so a session ID from
another user opens a new conversation rather than theirs.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from langgraph.graph.state import CompiledStateGraph
from sse_starlette import EventSourceResponse

from finai.agent.session import thread_id_for
from finai.api.deps import current_user, get_graph
from finai.api.schemas import ChatRequest
from finai.api.streaming import safe_answer_events

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest,
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
    user_id: Annotated[str, Depends(current_user)],
) -> EventSourceResponse:
    """Answer one question as a stream of events."""
    session_id = request.session_id or uuid.uuid4().hex

    async def stream() -> AsyncIterator:
        async for event in safe_answer_events(
            graph,
            request.message,
            user_id=user_id,
            thread_id=thread_id_for(user_id, session_id),
            session_id=session_id,
        ):
            yield event.to_sse()

    return EventSourceResponse(stream())
