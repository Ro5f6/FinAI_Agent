"""Health probes (2.1.5), at the root and without authentication.

* ``GET /healthz``: the process is alive. It checks nothing else, so an
  outage elsewhere never gets the API restarted.
* ``GET /readyz``: the API can serve requests: Qdrant (the 10-K index) and
  Redis (conversations, the fetch queue) both answer. Otherwise 503, so a
  load balancer sends traffic elsewhere until they are back.

The two checks run in parallel, each given ``CHECK_TIMEOUT_S``: a hung
dependency makes the probe fail, not hang. The clients are synchronous, so
each check runs in a worker thread.
"""

import asyncio
from collections.abc import Callable

from fastapi import APIRouter, Response, status

from finai.ingestion.jobs import redis_client
from finai.ingestion.store import qdrant_client

CHECK_TIMEOUT_S = 2.0

router = APIRouter(tags=["health"])


def check_qdrant() -> None:
    qdrant_client().get_collections()


def check_redis() -> None:
    redis_client().ping()


CHECKS: dict[str, Callable[[], None]] = {"qdrant": check_qdrant, "redis": check_redis}


async def _run(check: Callable[[], None]) -> str:
    try:
        await asyncio.wait_for(asyncio.to_thread(check), timeout=CHECK_TIMEOUT_S)
    except TimeoutError:
        return f"unreachable: no answer in {CHECK_TIMEOUT_S:g} s"
    except Exception as error:  # noqa: BLE001  any failure means not ready
        return f"unreachable: {type(error).__name__}"
    return "ok"


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """The process is alive."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(response: Response) -> dict[str, object]:
    """Qdrant and Redis both reachable, else 503."""
    names = list(CHECKS)
    results = await asyncio.gather(*(_run(CHECKS[name]) for name in names))
    checks = dict(zip(names, results, strict=True))
    ready = all(result == "ok" for result in results)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ready else "not_ready", "checks": checks}
