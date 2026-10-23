"""Dependencies the routes share.

The authenticated user (2.1.9: ``X-API-Key`` mapped to a ``user_id``) and
the compiled agent graph built at startup (2.1.4) are provided here, so a
route declares what it needs and tests can override it.
"""

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph


def get_graph(request: Request) -> CompiledStateGraph:
    """The agent graph the app built at startup."""
    return request.app.state.graph


def current_user() -> str:
    """The authenticated user. Until API-key auth (2.1.9) every caller is
    the demo user; routes depend on this, so only this function changes."""
    return "demo"
