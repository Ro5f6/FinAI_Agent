"""Tests for the ingestion worker.

Offline tests run ``process`` against a minimal stand-in for Redis, with the
pipeline and the Qdrant check replaced. ``-m redis`` runs the consumer group,
claiming and the whole loop against the compose Redis under test-only keys;
the pipeline is still replaced (1.3.30 ingests a real ticker end to end).
"""

import time

import pytest
import redis

from finai.config import settings
from finai.ingestion import jobs, worker
from finai.ingestion.fetch import TickerResolution
from finai.ingestion.ingest import IngestResult
from finai.ingestion.worker import Job, process

TTL_S = settings.INGEST_STATUS_TTL_MINUTES * 60


def _result(ticker: str) -> IngestResult:
    return IngestResult(
        ticker, "0000000000-26-000001", 2025, ("Item 1A",), {}, 42, 0, 1.0
    )


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.acked: list[str] = []
        self.history: list[str] = []  # statuses set, in order

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.values[key] = value
        self.ttls[key] = ex
        self.history.append(value)
        return True

    def delete(self, *keys: str) -> int:
        return sum(self.values.pop(k, None) is not None for k in keys)

    def xack(self, stream: str, group: str, *ids: str) -> int:
        self.acked.extend(ids)
        return len(ids)


@pytest.fixture
def fake() -> _FakeRedis:
    client = _FakeRedis()
    client.values["ingest:status:PLTR"] = "queued"
    return client


def test_success_clears_the_status_and_acknowledges(fake: _FakeRedis) -> None:
    calls = []

    def ingest(ticker, years):
        calls.append((ticker, years))
        assert fake.values["ingest:status:PLTR"] == "ingesting"  # set before the run
        return [_result(ticker)]

    outcome = process(
        fake, Job("1-0", "PLTR", 1), ingest=ingest, is_indexed=lambda t: False
    )

    assert outcome == "done"
    assert calls == [("PLTR", 1)]  # the CLI's default --years
    assert fake.history == ["ingesting"]
    assert fake.ttls["ingest:status:PLTR"] == TTL_S  # a fresh TTL, not KEEPTTL
    assert "ingest:status:PLTR" not in fake.values
    assert fake.acked == ["1-0"]


def test_failure_marks_failed_for_a_day_and_acknowledges(fake: _FakeRedis) -> None:
    def ingest(ticker, years):
        raise RuntimeError("Voyage is down")

    outcome = process(
        fake, Job("1-0", "PLTR", 1), ingest=ingest, is_indexed=lambda t: False
    )

    assert outcome == "failed"
    assert fake.history == ["ingesting", "failed"]
    assert fake.values["ingest:status:PLTR"] == "failed"
    assert fake.ttls["ingest:status:PLTR"] == TTL_S
    assert fake.acked == ["1-0"]


def test_already_indexed_ticker_is_skipped(fake: _FakeRedis) -> None:
    def ingest(ticker, years):
        raise AssertionError("ingested twice")

    outcome = process(
        fake, Job("1-0", "PLTR", 1), ingest=ingest, is_indexed=lambda t: True
    )

    assert outcome == "already_indexed"
    assert "ingest:status:PLTR" not in fake.values
    assert fake.acked == ["1-0"]


def test_a_job_that_keeps_killing_workers_is_given_up(fake: _FakeRedis) -> None:
    def ingest(ticker, years):
        raise AssertionError("run again")

    job = Job("1-0", "PLTR", worker.MAX_DELIVERIES + 1)
    outcome = process(fake, job, ingest=ingest, is_indexed=lambda t: False)

    assert outcome == "gave_up"
    assert fake.values["ingest:status:PLTR"] == "failed"
    assert fake.acked == ["1-0"]


def test_years_are_passed_through(fake: _FakeRedis) -> None:
    calls = []
    process(
        fake,
        Job("1-0", "PLTR", 1),
        years=3,
        ingest=lambda t, y: calls.append(y) or [],
        is_indexed=lambda t: False,
    )
    assert calls == [3]


def test_blocking_read_ends_before_the_socket_times_out() -> None:
    """A read blocked as long as the socket timeout raises TimeoutError."""
    from redis._defaults import DEFAULT_SOCKET_TIMEOUT

    assert worker.BLOCK_MS / 1000 < DEFAULT_SOCKET_TIMEOUT


class _FlakyRedis:
    """Fails the first read like a restarting Redis, then has no jobs."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.reads = 0
        self.groups_created = 0

    def xgroup_create(self, *args, **kwargs) -> bool:
        self.groups_created += 1
        return True

    def xautoclaim(self, *args, **kwargs):
        self.reads += 1
        if self.reads == 1:
            raise self.error
        return ["0-0", [], []]

    def xreadgroup(self, *args, **kwargs):
        return []


@pytest.mark.parametrize(
    "error", [redis.ConnectionError("refused"), redis.TimeoutError("timed out")]
)
def test_worker_survives_redis_dropping(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(worker, "RETRY_S", 0)
    client = _FlakyRedis(error)
    assert worker.run(once=True, client=client) == 0
    assert client.reads == 2  # retried, then found the queue empty


def test_worker_recreates_a_lost_group(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FlakyRedis(redis.ResponseError("NOGROUP No such key or consumer group"))
    worker.run(once=True, client=client)
    assert client.groups_created == 2  # at start, and after NOGROUP


def test_other_redis_errors_are_raised() -> None:
    client = _FlakyRedis(redis.ResponseError("WRONGTYPE"))
    with pytest.raises(redis.ResponseError, match="WRONGTYPE"):
        worker.run(once=True, client=client)


def test_cli_rejects_unknown_options() -> None:
    with pytest.raises(SystemExit):
        worker.main(["--bogus"])


# --- live Redis -------------------------------------------------------------

STREAM = "test:ingest:jobs"


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch) -> redis.Redis:
    """The compose Redis with the queue under test-only keys; fake pipeline."""
    client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    monkeypatch.setattr(jobs, "STREAM", STREAM)
    monkeypatch.setattr(jobs, "STATUS_PREFIX", "test:ingest:status:")
    monkeypatch.setattr(jobs, "COUNT_PREFIX", "test:ingest:count:")
    monkeypatch.setattr(jobs, "redis_client", lambda: client)
    monkeypatch.setattr(
        jobs,
        "resolve_ticker",
        lambda t: TickerResolution(ticker=t, status="ok", cik=1),
    )
    monkeypatch.setattr(worker, "_is_indexed", lambda t: False)
    monkeypatch.setattr(worker, "ingest_ticker", lambda t, y: [_result(t)])

    def clean() -> None:
        keys = list(client.scan_iter("test:ingest:*"))
        if keys:
            client.delete(*keys)

    clean()
    yield client
    clean()


@pytest.mark.redis
def test_live_worker_drains_the_queue(live: redis.Redis) -> None:
    for ticker in ("PLTR", "CRCL", "PLTR"):  # PLTR twice: queued once
        jobs.request_ingestion(ticker, user_id="alice")

    assert worker.run(once=True, client=live) == 2

    assert live.xpending(STREAM, worker.GROUP)["pending"] == 0
    assert not live.exists("test:ingest:status:PLTR", "test:ingest:status:CRCL")
    assert worker.run(once=True, client=live) == 0  # nothing left


@pytest.mark.redis
def test_live_failure_is_reported_to_the_next_question(
    live: redis.Redis, monkeypatch
) -> None:
    def broken(ticker, years):
        raise RuntimeError("parse error")

    monkeypatch.setattr(worker, "ingest_ticker", broken)
    jobs.request_ingestion("BAD", user_id="alice")
    worker.run(once=True, client=live)

    assert jobs.request_ingestion("BAD", user_id="alice").status == "failed"
    assert 0 < live.ttl("test:ingest:status:BAD") <= TTL_S
    assert live.xlen(STREAM) == 1  # not queued again while failed


@pytest.mark.redis
def test_live_job_of_a_dead_worker_is_claimed(
    live: redis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "CLAIM_IDLE_MS", 200)
    worker.ensure_group(live)
    jobs.request_ingestion("PLTR", user_id="alice")
    # A worker reads the job, then dies without acknowledging it.
    live.xreadgroup(worker.GROUP, "dead-worker", {STREAM: ">"}, count=1)
    time.sleep(0.3)

    job = worker.next_job(live, "survivor", block_ms=100)

    assert (job.ticker, job.deliveries) == ("PLTR", 2)
    assert worker.process(live, job) == "done"
    assert live.xpending(STREAM, worker.GROUP)["pending"] == 0


@pytest.mark.redis
def test_live_full_block_on_an_empty_queue(live: redis.Redis) -> None:
    """The long-running mode's read, which crashed with a 5 s block."""
    worker.ensure_group(live)
    started = time.monotonic()
    assert worker.next_job(live, "idle-worker") is None
    assert time.monotonic() - started >= worker.BLOCK_MS / 1000 * 0.9


@pytest.mark.redis
def test_live_group_creation_is_idempotent(live: redis.Redis) -> None:
    worker.ensure_group(live)
    worker.ensure_group(live)
    assert [g["name"] for g in live.xinfo_groups(STREAM)] == [worker.GROUP]
