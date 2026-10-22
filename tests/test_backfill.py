"""Tests for adding ciks to chunks stored before searches went by company."""

from qdrant_client import QdrantClient, models

from finai.ingestion.backfill import backfill_ciks
from finai.ingestion.fetch import TickerResolution
from finai.ingestion.store import ensure_collection, indexed_years

_DIM = 4


def _point(i: int, payload: dict) -> models.PointStruct:
    return models.PointStruct(id=i, vector=[1.0, 0.0, 0.0, 0.0], payload=payload)


def _resolution(ticker: str) -> TickerResolution:
    if ticker == "XOM":
        return TickerResolution(
            ticker="XOM", status="ok", cik=34088, current_cik=2115436
        )
    return TickerResolution(ticker=ticker, status="ok", cik={"AAPL": 320193}[ticker])


def _client() -> QdrantClient:
    client = QdrantClient(":memory:")
    ensure_collection(client, name="c", dim=_DIM)
    client.upsert(
        "c",
        [
            _point(1, {"ticker": "AAPL", "fiscal_year": 2025}),
            _point(2, {"ticker": "AAPL", "fiscal_year": 2024}),
            _point(3, {"ticker": "XOM", "fiscal_year": 2025}),
            _point(4, {"ticker": "CRCL", "fiscal_year": 2025, "ciks": [1876042]}),
        ],
    )
    return client


def test_adds_ciks_without_touching_other_points() -> None:
    client = _client()
    current = {"AAPL": 320193, "XOM": 2115436}.get

    written = backfill_ciks(client, "c", resolve=_resolution, current=current)

    assert written == {"AAPL": [320193], "XOM": [34088, 2115436]}
    assert indexed_years(client, 320193, name="c") == [2025, 2024]
    assert indexed_years(client, 2115436, name="c") == [2025]  # XOM by its new CIK
    assert indexed_years(client, 1876042, name="c") == [2025]  # already had ciks
    [point] = client.retrieve("c", [4], with_vectors=True)
    assert point.vector == [1.0, 0.0, 0.0, 0.0]  # not re-embedded


def test_running_again_changes_nothing() -> None:
    client = _client()
    current = {"AAPL": 320193, "XOM": 2115436}.get
    backfill_ciks(client, "c", resolve=_resolution, current=current)
    assert backfill_ciks(client, "c", resolve=_resolution, current=current) == {}
