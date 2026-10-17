"""Tests for queueing on-demand ingestion.

Offline tests replace Redis with a small in-memory stand-in and EDGAR with
fixed resolutions. ``-m redis`` runs the real Lua script against the compose
Redis, under test-only key names.
"""

import threading

import httpx
import pytest
import redis

from finai.config import settings
from finai.ingestion import jobs
from finai.ingestion.fetch import TickerResolution
from finai.ingestion.jobs import request_ingestion

_RESOLUTIONS = {
    "PLTR": TickerResolution(ticker="PLTR", status="ok", cik=1321655, name="Palantir"),
    "TSM": TickerResolution(
        ticker="TSM", status="no_10k", cik=1046179, message="TSM files Form 20-F"
    ),
    "ZZZZQ": TickerResolution(
        ticker="ZZZZQ", status="unknown_ticker", message="ZZZZQ is not a ticker"
    ),
}


class _FakeRedis:
    """get, set and the queue-once script, in memory."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.stream: list[dict[str, str]] = []
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise redis.ConnectionError("Redis is down")

    def get(self, key: str) -> str | None:
        self._check()
        return self.values.get(key)

    def register_script(self, source: str):
        def run(keys, args):
            self._check()
            if keys[0] in self.values:
                return None
            self.values[keys[0]] = args[0]
            self.ttls[keys[0]] = int(args[1])
            fields = dict(zip(("ticker", "cik", "requested_at"), args[3:], strict=True))
            self.stream.append({k: str(v) for k, v in fields.items()})
            return f"{len(self.stream)}-0"

        return run


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    client = _FakeRedis()
    resolved: list[str] = []

    def resolve(ticker: str) -> TickerResolution:
        resolved.append(ticker)
        return _RESOLUTIONS[ticker]

    monkeypatch.setattr(jobs, "redis_client", lambda: client)
    monkeypatch.setattr(jobs, "resolve_ticker", resolve)
    client.resolved = resolved
    return client


def test_a_new_ticker_is_queued_with_its_cik(fake: _FakeRedis) -> None:
    result = request_ingestion(" pltr ")

    assert (result.ticker, result.status) == ("PLTR", "queued")
    assert "PLTR's 10-K is queued" in result.message
    assert fake.values == {"ingest:status:PLTR": "queued"}
    assert fake.ttls["ingest:status:PLTR"] == settings.INGEST_STATUS_TTL_MINUTES * 60
    [job] = fake.stream
    assert (job["ticker"], job["cik"]) == ("PLTR", "1321655")
    assert job["requested_at"].endswith("+00:00")


def test_a_queued_ticker_is_not_queued_or_resolved_again(fake: _FakeRedis) -> None:
    request_ingestion("PLTR")
    again = request_ingestion("PLTR")

    assert again.status == "queued"
    assert len(fake.stream) == 1
    assert fake.resolved == ["PLTR"]  # the second call read the status only


@pytest.mark.parametrize("status", ["ingesting", "failed"])
def test_the_workers_status_is_reported(fake: _FakeRedis, status: str) -> None:
    fake.values["ingest:status:PLTR"] = status

    result = request_ingestion("PLTR")

    assert result.status == status
    assert fake.stream == []
    assert fake.resolved == []


@pytest.mark.parametrize("ticker", ["TSM", "ZZZZQ"])
def test_tickers_without_a_10k_are_not_queued(fake: _FakeRedis, ticker: str) -> None:
    result = request_ingestion(ticker)

    assert result.status == _RESOLUTIONS[ticker].status
    assert result.message == _RESOLUTIONS[ticker].message
    assert fake.values == {}
    assert fake.stream == []


def test_lost_race_reports_the_winners_status(fake: _FakeRedis) -> None:
    """Another replica queued it between this call's read and its script."""
    original = fake.register_script

    def racing(source: str):
        script = original(source)

        def run(keys, args):
            fake.values[keys[0]] = "ingesting"  # the other replica, then its worker
            return script(keys, args)

        return run

    fake.register_script = racing
    assert request_ingestion("PLTR").status == "ingesting"
    assert fake.stream == []


def test_redis_down_is_unavailable_not_an_error(fake: _FakeRedis) -> None:
    fake.fail = True
    result = request_ingestion("PLTR")
    assert result.status == "unavailable"
    assert "Redis is down" in result.message


def test_edgar_down_is_unavailable_and_nothing_is_queued(
    fake: _FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(ticker: str) -> TickerResolution:
        raise httpx.ConnectError("SEC down")

    monkeypatch.setattr(jobs, "resolve_ticker", down)
    result = request_ingestion("PLTR")
    assert result.status == "unavailable"
    assert fake.values == {}


# --- live Redis -------------------------------------------------------------


@pytest.fixture
def live_redis(monkeypatch: pytest.MonkeyPatch) -> redis.Redis:
    """The compose Redis, with the queue under test-only keys."""
    client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    monkeypatch.setattr(jobs, "STREAM", "test:ingest:jobs")
    monkeypatch.setattr(jobs, "STATUS_PREFIX", "test:ingest:status:")
    monkeypatch.setattr(jobs, "redis_client", lambda: client)
    monkeypatch.setattr(jobs, "resolve_ticker", lambda t: _RESOLUTIONS[t])
    client.delete("test:ingest:jobs", "test:ingest:status:PLTR")
    yield client
    client.delete("test:ingest:jobs", "test:ingest:status:PLTR")


@pytest.mark.redis
def test_live_queue_once_with_expiry(live_redis: redis.Redis) -> None:
    assert request_ingestion("PLTR").status == "queued"
    assert request_ingestion("PLTR").status == "queued"

    [(_, job)] = live_redis.xrange("test:ingest:jobs")
    assert (job["ticker"], job["cik"]) == ("PLTR", "1321655")
    ttl = live_redis.ttl("test:ingest:status:PLTR")
    assert (
        settings.INGEST_STATUS_TTL_MINUTES * 60 - 5
        < ttl
        <= settings.INGEST_STATUS_TTL_MINUTES * 60
    )


@pytest.mark.redis
def test_live_worker_status_change_keeps_the_expiry(live_redis: redis.Redis) -> None:
    request_ingestion("PLTR")
    live_redis.set("test:ingest:status:PLTR", "ingesting", keepttl=True)

    assert request_ingestion("PLTR").status == "ingesting"
    assert live_redis.ttl("test:ingest:status:PLTR") > 0


@pytest.mark.redis
def test_live_concurrent_requests_queue_once(live_redis: redis.Redis) -> None:
    results: list[str] = []
    barrier = threading.Barrier(20)

    def ask() -> None:
        barrier.wait()
        results.append(request_ingestion("PLTR").status)

    threads = [threading.Thread(target=ask) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == ["queued"] * 20
    assert live_redis.xlen("test:ingest:jobs") == 1
