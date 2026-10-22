"""Add ``ciks`` to chunks stored before searches went by company (D-043).

    python -m finai.ingestion.backfill

Chunks ingested before the field existed have no ``ciks`` and so can't be
found by a search, which now filters on it. For each ticker with such
chunks, this resolves the ticker's companies (the filer, and the company the
ticker maps to now, as the ingest does) and writes them into the existing
points with Qdrant's ``set_payload``: no re-embedding, no Voyage tokens.
Safe to repeat: points that already have ``ciks`` are left alone.
"""

import sys
from collections.abc import Callable

from qdrant_client import QdrantClient, models

from finai.config import settings
from finai.ingestion.fetch import TickerResolution, resolve_ticker, ticker_cik
from finai.ingestion.store import qdrant_client

_MISSING = models.IsEmptyCondition(is_empty=models.PayloadField(key="ciks"))


def _ticker_is(ticker: str) -> models.FieldCondition:
    return models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))


def tickers_missing_ciks(client: QdrantClient, name: str) -> list[str]:
    response = client.facet(
        collection_name=name,
        key="ticker",
        facet_filter=models.Filter(must=[_MISSING]),
        limit=10_000,
        exact=True,
    )
    return sorted(str(hit.value) for hit in response.hits)


def ciks_for(
    ticker: str,
    resolve: Callable[[str], TickerResolution],
    current: Callable[[str], int | None],
) -> list[int]:
    """The filer's CIK (the predecessor's, for XOM) and the ticker's now."""
    resolution = resolve(ticker)
    found = {resolution.cik, resolution.current_cik, current(ticker)}
    return sorted(cik for cik in found if cik is not None)


def backfill_ciks(
    client: QdrantClient | None = None,
    name: str | None = None,
    resolve: Callable[[str], TickerResolution] | None = None,
    current: Callable[[str], int | None] | None = None,
) -> dict[str, list[int]]:
    """Write ``ciks`` into every point lacking it; return ticker -> CIKs."""
    client = client or qdrant_client()
    name = name or settings.COLLECTION_NAME
    resolve = resolve or resolve_ticker
    current = current or ticker_cik
    written = {}
    for ticker in tickers_missing_ciks(client, name):
        ciks = ciks_for(ticker, resolve, current)
        if not ciks:
            continue  # SEC doesn't know the ticker; leave it unsearchable
        client.set_payload(
            collection_name=name,
            payload={"ciks": ciks},
            points=models.Filter(must=[_ticker_is(ticker), _MISSING]),
            wait=True,
        )
        written[ticker] = ciks
    return written


def main() -> int:
    written = backfill_ciks()
    for ticker, ciks in written.items():
        print(f"{ticker}: ciks {ciks}")
    print(f"{len(written)} tickers updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
