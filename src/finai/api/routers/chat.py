"""The chat endpoint: one question in, a streamed answer out (2.1.6).

Routes live under ``/api/v1``; the request and event shapes are in
``finai.api.schemas`` and the caller's identity comes from
``finai.api.deps``, never from the request body.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["chat"])
