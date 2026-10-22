"""Ingest the 10-Ks queued by questions about unindexed companies (1.3.28).

    python -m finai.ingestion.worker          # run until stopped
    python -m finai.ingestion.worker --once   # drain the queue, then exit

Reads ``ingest:jobs`` (1.3.27) through a consumer group, one job at a time,
which also keeps within SEC's request limit. Each job runs the 1.2.18
pipeline for the ticker's latest 10-K, then:

* success: the status key is deleted (the chunks now answer searches) and
  the job acknowledged;
* failure: the status becomes ``failed`` for a day, so the ticker isn't
  re-queued on every question, and the job is acknowledged. It is retried
  when the status expires and someone asks again.

A worker that dies mid-job leaves it unacknowledged. Another worker, or this
one restarted, claims it with ``XAUTOCLAIM`` once it has been idle for
``CLAIM_IDLE_MS``, longer than any ingest takes. A job delivered more than
``MAX_DELIVERIES`` times, one that keeps killing workers, is marked failed
instead of being run again.

Verified against redis-py 8.1.0 and Redis 8.10.2 (docs/CONTEXT.md).
``XREADGROUP ... CLAIM`` does reading and claiming in one call but is new in
Redis 8; ``XAUTOCLAIM`` works on any Redis from 6.2, managed ones included.

Status writes set a fresh TTL rather than ``KEEPTTL``: a job that waited
longer than the TTL finds its key gone, and ``SET ... KEEPTTL`` on a missing
key creates one that never expires, which would block the ticker for good.
"""

import argparse
import logging
import os
import signal
import socket
import sys
import time
from collections.abc import Callable
from typing import NamedTuple

import redis

from finai.config import settings
from finai.ingestion import jobs
from finai.ingestion.fetch import ticker_cik
from finai.ingestion.ingest import DEFAULT_YEARS, IngestResult, ingest_ticker
from finai.ingestion.store import indexed_years, qdrant_client

log = logging.getLogger(__name__)

GROUP = "ingest-workers"
# How long a read waits for a job; a stop request is noticed within this.
# It must stay under the client's socket timeout, 5 s by default in
# redis-py 8: a 5 s block on a 5 s socket timeout raised TimeoutError
# whenever the queue stayed empty.
BLOCK_MS = 2_000
# Pause before retrying after Redis drops or restarts.
RETRY_S = 2.0
# A job unacknowledged this long belongs to a dead worker. Ingesting one
# 10-K takes 25 s to a few minutes; 10 minutes leaves a wide margin.
CLAIM_IDLE_MS = 10 * 60 * 1000
MAX_DELIVERIES = 3


class Job(NamedTuple):
    id: str
    ticker: str
    deliveries: int


def ensure_group(client: redis.Redis) -> None:
    """Create the stream and consumer group if missing; safe to repeat."""
    try:
        client.xgroup_create(jobs.STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as error:
        if "BUSYGROUP" not in str(error):
            raise


def next_job(
    client: redis.Redis, consumer: str, block_ms: int = BLOCK_MS
) -> Job | None:
    """A job abandoned by a dead worker if any, else a new one, else None."""
    _, claimed, _ = client.xautoclaim(
        jobs.STREAM, GROUP, consumer, min_idle_time=CLAIM_IDLE_MS, count=1
    )
    if claimed:
        job_id, fields = claimed[0]
        [pending] = client.xpending_range(jobs.STREAM, GROUP, job_id, job_id, 1)
        return Job(job_id, fields["ticker"], pending["times_delivered"])

    read = client.xreadgroup(
        GROUP, consumer, {jobs.STREAM: ">"}, count=1, block=block_ms
    )
    if not read:
        return None
    job_id, fields = read[0][1][0]
    return Job(job_id, fields["ticker"], 1)


def _set_status(client: redis.Redis, ticker: str, status: jobs.JobStatus) -> None:
    client.set(
        jobs.status_key(ticker), status, ex=settings.INGEST_STATUS_TTL_MINUTES * 60
    )


def _is_indexed(ticker: str) -> bool:
    """Whether the ticker's company is stored, under any of its tickers."""
    cik = ticker_cik(ticker)
    return cik is not None and bool(indexed_years(qdrant_client(), cik))


def process(
    client: redis.Redis,
    job: Job,
    years: int = DEFAULT_YEARS,
    ingest: Callable[[str, int], list[IngestResult]] | None = None,
    is_indexed: Callable[[str], bool] | None = None,
) -> str:
    """Run one job and record the outcome; returns what happened.

    ``ingest`` and ``is_indexed`` default to the pipeline and the Qdrant
    check, looked up when called.
    """
    ingest = ingest or ingest_ticker
    is_indexed = is_indexed or _is_indexed
    if job.deliveries > MAX_DELIVERIES:
        log.error("%s: delivered %d times, giving up", job.ticker, job.deliveries)
        _set_status(client, job.ticker, "failed")
        client.xack(jobs.STREAM, GROUP, job.id)
        return "gave_up"

    # Queued twice (after the status expired) or ingested another way.
    if is_indexed(job.ticker):
        client.delete(jobs.status_key(job.ticker))
        client.xack(jobs.STREAM, GROUP, job.id)
        return "already_indexed"

    _set_status(client, job.ticker, "ingesting")
    try:
        results = ingest(job.ticker, years)
    # Any failure of one company's ingest must leave the worker running for
    # the next: EDGAR, parsing, Voyage and Qdrant each raise their own types.
    except Exception:
        log.exception("%s: ingest failed", job.ticker)
        _set_status(client, job.ticker, "failed")
        client.xack(jobs.STREAM, GROUP, job.id)
        return "failed"

    for result in results:
        log.info(result.summary())
    client.delete(jobs.status_key(job.ticker))
    client.xack(jobs.STREAM, GROUP, job.id)
    return "done"


def run(
    once: bool = False,
    years: int = DEFAULT_YEARS,
    client: redis.Redis | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> int:
    """Process jobs until stopped (or, with ``once``, until none are left).

    Returns the number of jobs processed.
    """
    client = client or jobs.redis_client()
    consumer = f"{socket.gethostname()}-{os.getpid()}"
    ensure_group(client)
    log.info("worker %s reading %s", consumer, jobs.STREAM)
    processed = 0
    while not should_stop():
        try:
            job = next_job(client, consumer, block_ms=100 if once else BLOCK_MS)
            if job is None:
                if once:
                    break
                continue
            log.info("%s: %s", job.ticker, process(client, job, years))
            processed += 1
        except (redis.ConnectionError, redis.TimeoutError) as error:
            # Redis restarting or the network dropping shouldn't end the
            # worker. A job cut off mid-way stays pending and is claimed later.
            log.warning("Redis unavailable, retrying in %.0f s: %s", RETRY_S, error)
            time.sleep(RETRY_S)
        except redis.ResponseError as error:
            # A Redis restarted without its data has lost the group.
            if "NOGROUP" not in str(error):
                raise
            log.warning("consumer group missing, recreating it")
            ensure_group(client)
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m finai.ingestion.worker",
        description="Ingest the 10-Ks queued for unindexed companies.",
    )
    parser.add_argument(
        "--once", action="store_true", help="drain the queue, then exit"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"10-Ks per company, newest first (default {DEFAULT_YEARS})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Finish the current job on SIGTERM (a pod being replaced) or Ctrl-C.
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.append(True))
    processed = run(once=args.once, years=args.years, should_stop=lambda: bool(stop))
    log.info("stopped after %d jobs", processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
