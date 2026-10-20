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


def _resolution(ticker: str) -> TickerResolution:
    """The fixed cases above; any other ticker is a 10-K filer."""
    return _RESOLUTIONS.get(ticker) or TickerResolution(
        ticker=ticker, status="ok", cik=1
    )


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
        # Mirrors _QUEUE_ONCE; the real script runs in the -m redis tests.
        def run(keys, args):
            self._check()
            status, _stream, user, total = keys
            if status in self.values:
                return [0, "exists"]
            if int(self.values.get(user, 0)) >= int(args[6]):
                return [0, "user_limit"]
            if int(self.values.get(total, 0)) >= int(args[7]):
                return [0, "total_limit"]
            self.values[status] = args[0]
            self.ttls[status] = int(args[1])
            for counter in (user, total):
                self.values[counter] = str(int(self.values.get(counter, 0)) + 1)
                self.ttls[counter] = int(args[8])
            fields = dict(
                zip(("ticker", "cik", "requested_at"), args[3:6], strict=True)
            )
            self.stream.append({k: str(v) for k, v in fields.items()})
            return [1, f"{len(self.stream)}-0"]

        return run


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    client = _FakeRedis()
    resolved: list[str] = []

    def resolve(ticker: str) -> TickerResolution:
        resolved.append(ticker)
        return _resolution(ticker)

    monkeypatch.setattr(jobs, "redis_client", lambda: client)
    monkeypatch.setattr(jobs, "resolve_ticker", resolve)
    client.resolved = resolved
    return client


def test_a_new_ticker_is_queued_with_its_cik(fake: _FakeRedis) -> None:
    result = request_ingestion(" pltr ", user_id="alice")

    assert (result.ticker, result.status) == ("PLTR", "queued")
    assert "PLTR's 10-K is queued" in result.message
    assert fake.values["ingest:status:PLTR"] == "queued"
    assert fake.ttls["ingest:status:PLTR"] == settings.INGEST_STATUS_TTL_MINUTES * 60
    [job] = fake.stream
    assert (job["ticker"], job["cik"]) == ("PLTR", "1321655")
    assert job["requested_at"].endswith("+00:00")


def test_a_queued_ticker_is_not_queued_or_resolved_again(fake: _FakeRedis) -> None:
    request_ingestion("PLTR", user_id="alice")
    again = request_ingestion("PLTR", user_id="alice")

    assert again.status == "queued"
    assert len(fake.stream) == 1
    assert fake.resolved == ["PLTR"]  # the second call read the status only


@pytest.mark.parametrize("status", ["ingesting", "failed"])
def test_the_workers_status_is_reported(fake: _FakeRedis, status: str) -> None:
    fake.values["ingest:status:PLTR"] = status

    result = request_ingestion("PLTR", user_id="alice")

    assert result.status == status
    assert fake.stream == []
    assert fake.resolved == []


@pytest.mark.parametrize("ticker", ["TSM", "ZZZZQ"])
def test_tickers_without_a_10k_are_not_queued(fake: _FakeRedis, ticker: str) -> None:
    result = request_ingestion(ticker, user_id="alice")

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
    assert request_ingestion("PLTR", user_id="alice").status == "ingesting"
    assert fake.stream == []


def test_redis_down_is_unavailable_not_an_error(fake: _FakeRedis) -> None:
    fake.fail = True
    result = request_ingestion("PLTR", user_id="alice")
    assert result.status == "unavailable"
    assert "Redis is down" in result.message


def test_edgar_down_is_unavailable_and_nothing_is_queued(
    fake: _FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(ticker: str) -> TickerResolution:
        raise httpx.ConnectError("SEC down")

    monkeypatch.setattr(jobs, "resolve_ticker", down)
    result = request_ingestion("PLTR", user_id="alice")
    assert result.status == "unavailable"
    assert fake.values == {}


# --- daily limits (1.3.29) ---------------------------------------------------


@pytest.fixture
def limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "INGEST_USER_DAILY_LIMIT", 2)
    monkeypatch.setattr(settings, "INGEST_DAILY_LIMIT", 3)


def test_a_user_is_stopped_at_their_daily_limit(fake: _FakeRedis, limits) -> None:
    assert request_ingestion("AAA", user_id="alice").status == "queued"
    assert request_ingestion("BBB", user_id="alice").status == "queued"

    third = request_ingestion("CCC", user_id="alice")

    assert third.status == "limit_reached"
    assert "your limit of 2 new companies a day" in third.message
    assert "midnight UTC" in third.message
    assert [j["ticker"] for j in fake.stream] == ["AAA", "BBB"]
    assert "ingest:status:CCC" not in fake.values  # not marked queued


def test_everyone_is_stopped_at_the_total_limit(fake: _FakeRedis, limits) -> None:
    for user, ticker in (("alice", "AAA"), ("alice", "BBB"), ("bob", "CCC")):
        assert request_ingestion(ticker, user_id=user).status == "queued"

    fourth = request_ingestion("DDD", user_id="carol")

    assert fourth.status == "limit_reached"
    assert "today's limit of 3 new companies" in fourth.message


def test_only_new_jobs_count(fake: _FakeRedis, limits) -> None:
    request_ingestion("AAA", user_id="alice")
    for _ in range(3):
        request_ingestion("AAA", user_id="alice")  # already queued
    request_ingestion("TSM", user_id="alice")  # files 20-F
    request_ingestion("ZZZZQ", user_id="alice")  # unknown

    user_key, total_key = jobs.count_keys(
        "alice", jobs.dt.datetime.now(jobs.dt.UTC).date()
    )
    assert (fake.values[user_key], fake.values[total_key]) == ("1", "1")
    assert request_ingestion("BBB", user_id="alice").status == "queued"


def test_a_queued_company_is_reported_even_past_the_limit(
    fake: _FakeRedis, limits
) -> None:
    """The limit stops new fetches, not news of one already under way."""
    request_ingestion("AAA", user_id="alice")
    request_ingestion("BBB", user_id="alice")
    assert request_ingestion("AAA", user_id="alice").status == "queued"


def test_limit_message_for_one_company(
    fake: _FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "INGEST_USER_DAILY_LIMIT", 1)
    request_ingestion("AAA", user_id="alice")
    message = request_ingestion("BBB", user_id="alice").message
    assert "your limit of 1 new company a day" in message


def test_counters_are_per_utc_day_and_expire() -> None:
    user_key, total_key = jobs.count_keys("alice", jobs.dt.date(2026, 10, 20))
    assert user_key == "ingest:count:user:alice:2026-10-20"
    assert total_key == "ingest:count:all:2026-10-20"


def test_counters_expire_after_two_days(fake: _FakeRedis) -> None:
    request_ingestion("AAA", user_id="alice")
    counters = [k for k in fake.ttls if k.startswith("ingest:count:")]
    assert len(counters) == 2
    assert all(fake.ttls[k] == 2 * 24 * 60 * 60 for k in counters)


# --- live Redis -------------------------------------------------------------


@pytest.fixture
def live_redis(monkeypatch: pytest.MonkeyPatch) -> redis.Redis:
    """The compose Redis, with the queue under test-only keys."""
    client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    monkeypatch.setattr(jobs, "STREAM", "test:ingest:jobs")
    monkeypatch.setattr(jobs, "STATUS_PREFIX", "test:ingest:status:")
    monkeypatch.setattr(jobs, "COUNT_PREFIX", "test:ingest:count:")
    monkeypatch.setattr(jobs, "redis_client", lambda: client)
    monkeypatch.setattr(jobs, "resolve_ticker", _resolution)

    def clean() -> None:
        keys = list(client.scan_iter("test:ingest:*"))
        if keys:
            client.delete(*keys)

    clean()
    yield client
    clean()


@pytest.mark.redis
def test_live_queue_once_with_expiry(live_redis: redis.Redis) -> None:
    assert request_ingestion("PLTR", user_id="alice").status == "queued"
    assert request_ingestion("PLTR", user_id="alice").status == "queued"

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
    request_ingestion("PLTR", user_id="alice")
    live_redis.set("test:ingest:status:PLTR", "ingesting", keepttl=True)

    assert request_ingestion("PLTR", user_id="alice").status == "ingesting"
    assert live_redis.ttl("test:ingest:status:PLTR") > 0


@pytest.mark.redis
def test_live_concurrent_requests_queue_once(live_redis: redis.Redis) -> None:
    results: list[str] = []
    barrier = threading.Barrier(20)

    def ask() -> None:
        barrier.wait()
        results.append(request_ingestion("PLTR", user_id="alice").status)

    threads = [threading.Thread(target=ask) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == ["queued"] * 20
    assert live_redis.xlen("test:ingest:jobs") == 1


@pytest.mark.redis
def test_live_limits_hold_under_concurrent_requests(
    live_redis: redis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """20 new companies at once from one user: exactly the limit is queued."""
    monkeypatch.setattr(settings, "INGEST_USER_DAILY_LIMIT", 5)
    results: list[str] = []
    barrier = threading.Barrier(20)

    def ask(ticker: str) -> None:
        barrier.wait()
        results.append(request_ingestion(ticker, user_id="alice").status)

    threads = [threading.Thread(target=ask, args=(f"T{i:02d}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == ["limit_reached"] * 15 + ["queued"] * 5
    assert live_redis.xlen("test:ingest:jobs") == 5
    [user_key] = live_redis.keys("test:ingest:count:user:alice:*")
    assert live_redis.get(user_key) == "5"
    assert 0 < live_redis.ttl(user_key) <= 2 * 24 * 60 * 60


@pytest.mark.redis
def test_live_total_limit_spans_users(
    live_redis: redis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "INGEST_DAILY_LIMIT", 2)
    assert request_ingestion("AAA", user_id="alice").status == "queued"
    assert request_ingestion("BBB", user_id="bob").status == "queued"
    assert request_ingestion("CCC", user_id="carol").status == "limit_reached"
    assert live_redis.xlen("test:ingest:jobs") == 2
