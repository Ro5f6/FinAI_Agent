"""Request and response shapes of the API (2.1.3).

``ChatRequest`` is what a client sends; ``StreamEvent`` is each server-sent
event of the answer stream. Neither carries a ``user_id``: the user is
always taken from authentication (2.1.9), and a request naming one is
rejected rather than ignored.

``session_id`` picks the conversation to continue. It becomes part of a
storage key, so it is limited to letters, digits, ``_`` and ``-``; and it
is never the thread ID by itself: the thread is namespaced by the
authenticated user (2.1.7), so a session ID guessed or copied from someone
else opens a new conversation of the caller's own, not theirs.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sse_starlette.event import JSONServerSentEvent

from finai.config import settings

EventType = Literal["token", "tool_start", "tool_end", "error", "done"]

SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class ChatRequest(BaseModel):
    """One question, optionally continuing a conversation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(
        min_length=1,
        max_length=settings.CHAT_MAX_MESSAGE_CHARS,
        description="The question, in plain words.",
        examples=["What's NVDA trading at?"],
    )
    session_id: SessionId | None = Field(
        default=None,
        description=(
            "The conversation to continue. Omit it to start a new one; the "
            "server's `done` event says which session the answer belongs to."
        ),
        examples=["3f2b9c1e-portfolio"],
    )


class StreamEvent(BaseModel):
    """One event of the answer stream.

    ``type`` is the SSE event name; ``data`` its JSON payload. Shapes, as the
    chat endpoint sends them (2.1.6): ``token`` {"text"}, ``tool_start``
    {"tool", "args", "call_id"}, ``tool_end`` {"tool", "call_id", "status"},
    ``error`` {"message"}, ``done`` {"session_id"}.
    """

    model_config = ConfigDict(frozen=True)

    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> JSONServerSentEvent:
        """The event as sse-starlette sends it: ``event: <type>``, JSON data."""
        return JSONServerSentEvent(data=self.data, event=self.type)
