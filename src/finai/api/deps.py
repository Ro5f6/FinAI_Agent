"""Dependencies the routes share.

The authenticated user (2.1.9: ``X-API-Key`` mapped to a ``user_id``) and
the compiled agent graph built at startup (2.1.4) are provided here, so a
route declares what it needs and tests can override it.
"""

import hmac
import logging
from typing import Annotated

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from langgraph.graph.state import CompiledStateGraph

from finai.config import settings
from finai.config.settings import api_key_users

log = logging.getLogger(__name__)


def get_graph(request: Request) -> CompiledStateGraph:
    """The agent graph the app built at startup."""
    return request.app.state.graph


# Missing header: FastAPI answers 401 with "WWW-Authenticate: APIKey" itself
# (fastapi 0.161.1, APIKeyHeader.make_not_authenticated_error).
api_key_header = APIKeyHeader(
    name="X-API-Key",
    description="Your API key; it identifies you, so no request carries a user ID.",
)


def _invalid_key() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "APIKey"},
    )


def current_user(api_key: Annotated[str, Security(api_key_header)]) -> str:
    """The user the ``X-API-Key`` header belongs to (2.1.9), else 401.

    Every configured key is compared in constant time, so response timing
    reveals nothing about how much of a key was right.
    """
    users = api_key_users(settings.API_KEYS)
    if not users:
        log.warning("chat request refused: no API_KEYS configured")
        raise _invalid_key()
    user_id = None
    for key, user in users.items():
        if hmac.compare_digest(api_key.encode(), key.encode()):
            user_id = user
    if user_id is None:
        raise _invalid_key()
    return user_id
