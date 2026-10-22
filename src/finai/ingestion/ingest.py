"""Ingest companies' recent 10-Ks into Qdrant.

    python -m finai.ingestion.ingest --ticker AAPL NVDA MSFT --years 3
    python -m finai.ingestion.ingest --tickers-file reference/sp500-2026-10-20.csv

Fetch, archive, extract, clean, chunk, embed and upsert, in that order; each
step is the module of the same name. Re-running is safe: chunk IDs are
deterministic, so a filing's points are overwritten, and ``upsert_filing``
checks the stored count against the chunks written.

``--tickers-file`` ingests a universe (1.3.31): the latest 10-K of each
ticker, one filing at a time. A filing whose accession is already stored is
skipped, so an interrupted run resumes where it stopped, and a share-class
sibling (GOOG after GOOGL: one filing) isn't embedded twice. A ticker that
fails is logged and the run continues; the summary lists every failure.
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

from edgar.entity import EntityFiling

from finai.ingestion.chunk import chunk_filing
from finai.ingestion.clean import clean_items
from finai.ingestion.embed import embed_documents
from finai.ingestion.fetch import (
    archive_filing,
    company_ciks,
    filing_meta,
    get_latest_10ks,
)
from finai.ingestion.parse import extract_filing_items
from finai.ingestion.store import (
    ensure_collection,
    ensure_payload_indexes,
    filing_stored,
    qdrant_client,
    upsert_filing,
)
from finai.ingestion.universe import read_tickers

log = logging.getLogger(__name__)

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


class NoItemsError(LookupError):
    """No 10-K item could be extracted, so nothing would be stored.

    Six S&P 500 filings in the 1.3.32 run: edgartools found no sections in
    C, EMR, GWW, JBHT and YUM, and WFC's report is in an EX-13 exhibit. Raised
    rather than storing 0 chunks as a success, which would let the on-demand
    queue re-ingest the company on every question.
    """


def ingest_filing(filing: EntityFiling, ticker: str) -> IngestResult:
    """Run one filing through every step and store its chunks.

    Raises:
        NoItemsError: no item had text to chunk.
    """
    started = time.monotonic()
    meta = filing_meta(filing, ticker)._replace(ciks=company_ciks(filing, ticker))
    archive_filing(filing, ticker)

    cleaned = clean_items(extract_filing_items(filing))
    if not cleaned.sections:
        raise NoItemsError(
            f"{meta.ticker} {meta.accession}: no 10-K items could be extracted"
        )
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


class UniverseRun(NamedTuple):
    """What a ``--tickers-file`` run did."""

    ingested: list[IngestResult]
    skipped: list[str]  # filing already stored
    failed: dict[str, str]  # ticker -> reason
    seconds: float

    def summary(self) -> str:
        chunks = sum(r.chunks for r in self.ingested)
        headline = (
            f"{len(self.ingested)} ingested ({chunks} chunks), "
            f"{len(self.skipped)} already stored, {len(self.failed)} failed "
            f"in {self.seconds / 60:.1f} min"
        )
        lines = [headline]
        lines += [f"  failed {t}: {reason}" for t, reason in self.failed.items()]
        return "\n".join(lines)


def ingest_universe(
    tickers: Sequence[str],
    on_progress: Callable[[str], None] = lambda line: None,
) -> UniverseRun:
    """Ingest the latest 10-K of each ticker, skipping filings already stored."""
    started = time.monotonic()
    client = qdrant_client()
    ingested: list[IngestResult] = []
    skipped: list[str] = []
    failed: dict[str, str] = {}
    for i, ticker in enumerate(tickers, 1):
        prefix = f"[{i}/{len(tickers)}] {ticker}"
        try:
            [filing] = get_latest_10ks(ticker, 1)
            if filing_stored(client, filing.accession_no):
                skipped.append(ticker)
                on_progress(f"{prefix}: {filing.accession_no} already stored")
                continue
            result = ingest_filing(filing, ticker)
        # One company failing must not end a run of 500: EDGAR, parsing,
        # Voyage and Qdrant each raise their own types. Logged and listed.
        except Exception as error:
            failed[ticker] = f"{type(error).__name__}: {error}"
            log.warning(
                "%s failed", prefix, exc_info=not isinstance(error, LookupError)
            )
            on_progress(f"{prefix}: FAILED {failed[ticker]}")
            continue
        ingested.append(result)
        on_progress(f"{prefix}: {result.summary()}")
    return UniverseRun(ingested, skipped, failed, time.monotonic() - started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m finai.ingestion.ingest",
        description="Ingest companies' recent 10-Ks into Qdrant.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ticker", nargs="+", help="one or more, e.g. AAPL NVDA")
    source.add_argument(
        "--tickers-file",
        type=Path,
        help="a universe CSV (reference/sp500-*.csv): latest 10-K each, resumable",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help=f"10-Ks per company, newest first (default {DEFAULT_YEARS})",
    )
    args = parser.parse_args(argv)
    if args.tickers_file is not None:
        if args.years is not None:
            parser.error("--tickers-file ingests the latest 10-K only; drop --years")
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
        run = ingest_universe(
            read_tickers(args.tickers_file),
            on_progress=lambda line: print(line, flush=True),
        )
        print(run.summary(), flush=True)
        return 1 if run.failed else 0

    args.years = DEFAULT_YEARS if args.years is None else args.years
    if args.years < 1:
        parser.error("--years must be at least 1")

    failed = False
    for ticker in args.ticker:
        try:
            results = ingest_ticker(ticker, args.years)
        except LookupError as error:
            # UnknownTickerError, FilingNotFoundError (fetch.resolve_ticker) and
            # NoItemsError are LookupErrors: an unknown ticker, a company with
            # no 10-K, or a 10-K whose items couldn't be extracted.
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
