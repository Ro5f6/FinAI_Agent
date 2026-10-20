"""Ingest companies' recent 10-Ks into Qdrant.

    python -m finai.ingestion.ingest --ticker AAPL NVDA MSFT --years 3

Fetch, archive, extract, clean, chunk, embed and upsert, in that order; each
step is the module of the same name. Re-running is safe: chunk IDs are
deterministic, so a filing's points are overwritten, and ``upsert_filing``
checks the stored count against the chunks written.
"""

import argparse
import sys
import time
from typing import NamedTuple

from edgar.entity import EntityFiling

from finai.ingestion.chunk import chunk_filing
from finai.ingestion.clean import clean_items
from finai.ingestion.embed import embed_documents
from finai.ingestion.fetch import archive_filing, filing_meta, get_latest_10ks
from finai.ingestion.parse import extract_filing_items
from finai.ingestion.store import (
    ensure_collection,
    ensure_payload_indexes,
    qdrant_client,
    upsert_filing,
)

# 10-Ks per company when the CLI or the worker (1.3.28) isn't told otherwise.
DEFAULT_YEARS = 1


class IngestResult(NamedTuple):
    """What one filing's ingest produced."""

    ticker: str
    accession: str
    fiscal_year: int
    sections: tuple[str, ...]
    # Items dropped as cross-references, with their cleaned length (D-016).
    vestigial: dict[str, int]
    chunks: int
    stale_removed: int
    seconds: float

    def summary(self) -> str:
        dropped = (
            ", dropped "
            + ", ".join(f"{k} ({v} chars)" for k, v in self.vestigial.items())
            if self.vestigial
            else ""
        )
        return (
            f"{self.ticker} FY{self.fiscal_year} {self.accession}: "
            f"{self.chunks} chunks from {', '.join(self.sections)}{dropped}; "
            f"{self.stale_removed} stale removed; {self.seconds:.1f}s"
        )


def ingest_filing(filing: EntityFiling, ticker: str) -> IngestResult:
    """Run one filing through every step and store its chunks."""
    started = time.monotonic()
    meta = filing_meta(filing, ticker)
    archive_filing(filing, ticker)

    cleaned = clean_items(extract_filing_items(filing))
    payloads = chunk_filing(meta, cleaned.sections)
    vectors = embed_documents([p.text for p in payloads])

    client = qdrant_client()
    ensure_collection(client)
    ensure_payload_indexes(client)
    removed = upsert_filing(client, payloads, vectors)

    return IngestResult(
        ticker=meta.ticker,
        accession=meta.accession,
        fiscal_year=meta.fiscal_year,
        sections=tuple(cleaned.sections),
        vestigial=dict(cleaned.vestigial),
        chunks=len(payloads),
        stale_removed=removed,
        seconds=time.monotonic() - started,
    )


def ingest_ticker(ticker: str, years: int = DEFAULT_YEARS) -> list[IngestResult]:
    """Ingest the latest ``years`` 10-Ks for ``ticker``, newest first.

    Each year has its own accession, so its chunks get their own IDs and the
    years don't overwrite each other.
    """
    return [ingest_filing(f, ticker) for f in get_latest_10ks(ticker, years)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m finai.ingestion.ingest",
        description="Ingest companies' recent 10-Ks into Qdrant.",
    )
    parser.add_argument(
        "--ticker", required=True, nargs="+", help="one or more, e.g. AAPL NVDA"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"10-Ks per company, newest first (default {DEFAULT_YEARS})",
    )
    args = parser.parse_args(argv)
    if args.years < 1:
        parser.error("--years must be at least 1")

    failed = False
    for ticker in args.ticker:
        try:
            results = ingest_ticker(ticker, args.years)
        except LookupError as error:
            # UnknownTickerError and FilingNotFoundError (fetch.resolve_ticker) are
            # both LookupErrors: an unknown ticker, or a company with no 10-K.
            # The other tickers still run.
            print(f"error: {error}", file=sys.stderr)
            failed = True
            continue
        for result in results:
            print(result.summary(), flush=True)
        if len(results) < args.years:
            print(
                f"{ticker.upper()}: only {len(results)} of {args.years} years exist",
                flush=True,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
