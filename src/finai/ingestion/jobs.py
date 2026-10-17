"""Queue on-demand ingestion of a company's 10-K (1.3.27).

When a question names a company with no 10-K in Qdrant, the agent answers
with the tools that work for any ticker and the filing is queued here, so a
later question can cite it. A separate worker (1.3.28) runs the ingestion;
it takes 25 s or more, too long to hold a request open.

Two Redis keys per ticker, verified against redis-py 8.1.0 and the compose
Redis 8 (docs/CONTEXT.md):

* ``ingest:status:{TICKER}`` holds ``queued``, ``ingesting`` or ``failed``
  and expires after ``INGEST_STATUS_TTL_MINUTES``. It is also the "queue
  once" guard: it is created with ``SET NX``, so of two API replicas, or two
  questions in a row, only the first queues the ticker. The worker changes it
  with ``SET ... KEEPTTL`` and deletes it on success, when the chunks
  themselves show the filing is there.
* ``ingest:jobs`` is the Redis Stream the worker reads.

``SET NX`` and ``XADD`` run in one Lua script, so a crash between them can't
leave a ticker marked queued with no job behind it for a day.
"""

import datetime as dt
from functools import lru_cache
from typing import Literal

import httpx
import redis
from edgar.exceptions import EdgarError
from pydantic import BaseModel, ConfigDict

from finai.config import settings
from finai.ingestion.fetch import resolve_ticker

STREAM = "ingest:jobs"
STATUS_PREFIX = "ingest:status:"
# Jobs are acknowledged by the worker; older entries are trimmed past this
# many (approximately, which lets Redis trim whole nodes cheaply).
STREAM_MAXLEN = 10_000

JobStatus = Literal["queued", "ingesting", "failed"]
RequestStatus = Literal[
    "queued", "ingesting", "failed", "no_10k", "unknown_ticker", "unavailable"
]

_MESSAGES: dict[str, str] = {
    "queued": "{t}'s 10-K is queued for fetching; it can be searched in a few minutes.",
    "ingesting": "{t}'s 10-K is being fetched and indexed; it can be searched shortly.",
    "failed": "Fetching {t}'s 10-K failed; it will be retried later.",
}

# KEYS[1] status key, KEYS[2] stream. ARGV: status, ttl seconds, maxlen,
# then the job's fields. Returns the stream ID, or nil if already queued.
_QUEUE_ONCE = """
if redis.call('SET', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[2]) then
  return redis.call('XADD', KEYS[2], 'MAXLEN', '~', ARGV[3], '*',
                    'ticker', ARGV[4], 'cik', ARGV[5], 'requested_at', ARGV[6])
end
return false
"""


class IngestionRequest(BaseModel):
    """What happened to a request to ingest a ticker's 10-K."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    status: RequestStatus
    message: str


def status_key(ticker: str) -> str:
    return f"{STATUS_PREFIX}{ticker}"


@lru_cache(maxsize=1)
def redis_client() -> redis.Redis:
    """One connection pool per process, strings in and out."""
    return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _request(ticker: str, status: RequestStatus, message: str) -> IngestionRequest:
    return IngestionRequest(ticker=ticker, status=status, message=message)


def request_ingestion(ticker: str) -> IngestionRequest:
    """Queue ``ticker``'s latest 10-K for ingestion, once.

    Returns the ticker's current status if it is already queued, being
    ingested or failed recently; otherwise resolves it (1.3.26) and queues
    it, or says why it can't be: ``no_10k`` (a 20-F filer, an ETF, a fund)
    or ``unknown_ticker``. ``unavailable`` means Redis or EDGAR couldn't be
    reached; nothing was queued.
    """
    symbol = ticker.strip().upper()
    client = redis_client()
    try:
        current = client.get(status_key(symbol))
    except redis.RedisError as error:
        return _request(
            symbol, "unavailable", f"The fetch queue is unavailable: {error}"
        )
    if current in _MESSAGES:
        return _request(symbol, current, _MESSAGES[current].format(t=symbol))

    # Resolving costs EDGAR requests, so it happens only when the ticker
    # isn't already in the queue.
    try:
        resolution = resolve_ticker(symbol)
    except (httpx.HTTPError, EdgarError) as error:
        return _request(
            symbol, "unavailable", f"SEC EDGAR could not be reached: {error}"
        )
    if resolution.status != "ok":
        return _request(symbol, resolution.status, resolution.message or "")

    try:
        queued = client.register_script(_QUEUE_ONCE)(
            keys=[status_key(symbol), STREAM],
            args=[
                "queued",
                settings.INGEST_STATUS_TTL_MINUTES * 60,
                STREAM_MAXLEN,
                symbol,
                resolution.cik,
                dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            ],
        )
        # Another replica may have queued it between the read and the script.
        current = "queued" if queued else client.get(status_key(symbol)) or "queued"
    except redis.RedisError as error:
        return _request(
            symbol, "unavailable", f"The fetch queue is unavailable: {error}"
        )
    return _request(symbol, current, _MESSAGES[current].format(t=symbol))
