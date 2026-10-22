"""On-demand ingestion end to end (1.3.30).

A question about a company with no 10-K stored queues it; the worker fetches
it from SEC, embeds it with Voyage and stores it in Qdrant; the same question
then returns passages. Everything is real except where it is kept: a
throwaway Qdrant collection, a temporary archive folder and ``test:`` Redis
keys, all removed afterwards.

    uv run pytest tests/test_ondemand.py -m "network and qdrant and voyage and redis"

Costs one 10-K of Voyage tokens (Figma's: about 100k tokens, under a cent).
"""

from pathlib import Path

import pytest
import redis

from finai.config import settings
from finai.ingestion import jobs, worker
from finai.ingestion.fetch import ticker_cik
from finai.ingestion.store import indexed_years, qdrant_client
from finai.tools.agent_tools import query_company_10k

TICKER = "FIG"  # Figma: one 10-K (FY2025) on EDGAR
COLLECTION = "test_ondemand_e2e"
USER = {"configurable": {"user_id": "e2e-test"}}
QUESTION = "What risks does Figma describe around AI?"


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> redis.Redis:
    client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    monkeypatch.setattr(settings, "COLLECTION_NAME", COLLECTION)
    monkeypatch.setattr(settings, "RAW_FILINGS_DIR", tmp_path)
    monkeypatch.setattr(jobs, "STREAM", "test:ingest:jobs")
    monkeypatch.setattr(jobs, "STATUS_PREFIX", "test:ingest:status:")
    monkeypatch.setattr(jobs, "COUNT_PREFIX", "test:ingest:count:")
    monkeypatch.setattr(jobs, "redis_client", lambda: client)

    def clean() -> None:
        keys = list(client.scan_iter("test:ingest:*"))
        if keys:
            client.delete(*keys)
        if qdrant_client().collection_exists(COLLECTION):
            qdrant_client().delete_collection(COLLECTION)

    clean()
    yield client
    clean()


def _ask() -> dict:
    return query_company_10k.invoke(
        {"ticker": TICKER, "question": QUESTION}, config=USER
    )


@pytest.mark.network
@pytest.mark.qdrant
@pytest.mark.voyage
@pytest.mark.redis
def test_unindexed_company_is_fetched_and_then_searchable(
    isolated: redis.Redis,
) -> None:
    # The collection doesn't exist yet: the first question creates nothing
    # in Qdrant, only the job.
    isolated_collection_missing = not qdrant_client().collection_exists(COLLECTION)
    assert isolated_collection_missing

    first = _ask()
    assert first["status"] == "queued", first
    assert first["passages"] == []
    assert isolated.xlen("test:ingest:jobs") == 1

    assert worker.run(once=True, client=isolated) == 1

    # Done: the status key is gone, the job acknowledged, the chunks stored.
    assert not isolated.exists(f"test:ingest:status:{TICKER}")
    assert isolated.xpending("test:ingest:jobs", worker.GROUP)["pending"] == 0
    [year] = indexed_years(qdrant_client(), ticker_cik(TICKER))  # by company
    assert year == 2025
    assert any(settings.RAW_FILINGS_DIR.joinpath(TICKER).iterdir())  # archived

    second = _ask()
    assert second["status"] == "ok", second
    assert second["fiscal_year"] == year
    assert len(second["passages"]) == 5
    for passage in second["passages"]:
        assert passage["ticker"] == TICKER
        assert passage["fiscal_year"] == year
        assert passage["accession"]
        assert passage["text"]

    # Asking again doesn't queue it again or use more of the user's quota.
    _ask()
    assert isolated.xlen("test:ingest:jobs") == 1
    [counter] = isolated.keys("test:ingest:count:user:e2e-test:*")
    assert isolated.get(counter) == "1"
