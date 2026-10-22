"""Fetching and archiving SEC filings.

Two separate concerns, deliberately kept apart:

* The **cache** is edgartools' own local store (``EDGAR_LOCAL_DATA_DIR``). It is
  what makes re-runs free and parsing offline: once a filing is in the store,
  ``filing.obj()`` reads from disk and the SEC is not contacted again.
* The **archive** is this module's gzipped copy under ``RAW_FILINGS_DIR``. It
  exists for provenance -- being able to open the exact document behind a cited
  claim -- not for performance, and nothing downstream reads it.

HTML compresses about 14x, so the archive costs roughly 110 KB per filing
rather than 1.5 MB.

Tickers are resolved to a CIK before anything is fetched (1.3.26), from two
tables (verified 2026-10-17, edgartools 5.60.0):

* edgartools' bundled table, frozen at its release. ``find_company_cik``
  prefers it and asks SEC only for tickers it lacks, so a ticker that moved
  to a new CIK keeps the old one.
* SEC's live ``company_tickers.json``. It differed from the bundled table on
  ten tickers' CIKs. Eight are reorganisations into a new registrant with no
  10-K yet, the 10-Ks sitting under the predecessor: XOM is now
  ExxonMobil Holdings Corp (CIK 2115436, no 10-K), and its 31 10-Ks are
  under Exxon Mobil Corp (34088). One (UROY) already filed under the new CIK.

So the live table's CIK comes first, and the bundled table's CIK is used
only if the live one has no 10-K and the bundled one filed a 10-K recently.
The recency test keeps a ticker that was reused by another company, or one
that survives only in the bundled table after delisting, from resolving to
a dead company's old filings.
"""

import datetime as dt
import gzip
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Literal, NamedTuple
from zoneinfo import ZoneInfo

import edgar
import httpx
from edgar import Company, get_identity, set_identity
from edgar.entity import EntityFiling
from edgar.exceptions import EdgarError
from edgar.httprequests import download_json
from edgar.reference.tickers import find_mutual_fund_cik, get_company_cik_lookup
from edgar.urls import build_company_tickers_url
from pydantic import BaseModel, ConfigDict

from finai.config import settings

log = logging.getLogger(__name__)

NEW_YORK = ZoneInfo("America/New_York")

# SEC's ticker table changes as companies list, delist and reorganise; a day
# is fresh enough and costs one 850 KB download.
SEC_TICKERS_TTL_S = 24 * 60 * 60
# A predecessor CIK is used only if its latest 10-K was filed this recently:
# a company that still files annually has filed within 15 months. XOM's
# predecessor filed on 2026-02-18.
PREDECESSOR_MAX_AGE = dt.timedelta(days=456)
FOREIGN_ANNUAL_FORMS = ("20-F", "40-F")


class FilingNotFoundError(LookupError):
    """No filing of the requested form exists for the requested company."""


class UnknownTickerError(LookupError):
    """SEC has no company or fund under the ticker."""


class UnsupportedFilingError(TypeError):
    """The filing yielded no usable HTML document."""


def configure_edgar_identity() -> str:
    """Register the SEC contact identity for this process.

    Returns the identity edgartools will actually send, read back rather than
    echoed, so a caller sees what was registered and not merely what was asked
    for.
    """
    set_identity(settings.SEC_IDENTITY)
    return get_identity()


def configure_edgar() -> None:
    """Register the identity, the rate limit and the local store.

    The rate limit is read by edgartools from the environment at request time,
    so it has to be exported rather than passed as an argument. ``Settings``
    remains the single source of the value.
    """
    configure_edgar_identity()

    os.environ["EDGAR_RATE_LIMIT_PER_SEC"] = str(settings.EDGAR_RATE_LIMIT_PER_SEC)

    # use_local_storage raises FileNotFoundError on a missing directory.
    settings.EDGAR_LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    edgar.use_local_storage(
        settings.EDGAR_LOCAL_DATA_DIR,
        allow_network_fallback=settings.EDGAR_ALLOW_NETWORK_FALLBACK,
    )


class TickerResolution(BaseModel):
    """What a ticker refers to on EDGAR, and whether it has a 10-K."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    status: Literal["ok", "unknown_ticker", "no_10k"]
    # The registrant whose 10-Ks to read; for no_10k, the one checked.
    cik: int | None = None
    name: str | None = None
    # Set when the ticker's current registrant has no 10-K yet and the 10-Ks
    # are its predecessor's (cik): XOM after its holding-company reorganisation.
    current_cik: int | None = None
    message: str | None = None


_sec_tickers: tuple[float, dict[str, int]] | None = None
_sec_tickers_lock = threading.Lock()


def _ticker_key(symbol: str) -> str:
    """SEC writes share classes with a dash: BRK.B is BRK-B."""
    return symbol.replace(".", "-")


def _live_ciks() -> dict[str, int] | None:
    """SEC's current ticker -> CIK table, refreshed daily; None if unreachable."""
    global _sec_tickers
    with _sec_tickers_lock:
        if _sec_tickers and time.monotonic() - _sec_tickers[0] < SEC_TICKERS_TTL_S:
            return _sec_tickers[1]
    try:
        rows = download_json(build_company_tickers_url()).values()
    # httpx failures, edgartools' rate-limit and transport errors, bad JSON.
    except (httpx.HTTPError, EdgarError, ValueError) as error:
        log.warning(
            "SEC ticker table unavailable, using edgartools' bundled one: %s", error
        )
        return _sec_tickers[1] if _sec_tickers else None
    table: dict[str, int] = {}
    for row in rows:
        table.setdefault(str(row["ticker"]).upper(), int(row["cik_str"]))
    with _sec_tickers_lock:
        _sec_tickers = (time.monotonic(), table)
    return table


def _bundled_cik(symbol: str) -> int | None:
    """The CIK in edgartools' bundled table, without its live fallback."""
    cik = get_company_cik_lookup().get(_ticker_key(symbol))
    return int(cik) if cik is not None else None


def ticker_cik(ticker: str) -> int | None:
    """The company a ticker belongs to now, from cached tables: no SEC request.

    SEC's live table first (downloaded once a day), then edgartools' bundled
    one. Share classes map to one company: GOOGL and GOOG are both 1652044.
    Searches go by this CIK, not the ticker (D-043).
    """
    symbol = ticker.strip().upper()
    live = _live_ciks()
    if live is not None and (cik := live.get(_ticker_key(symbol))) is not None:
        return cik
    return _bundled_cik(symbol)


def company_ciks(filing: EntityFiling, ticker: str) -> tuple[int, ...]:
    """The CIKs a filing's chunks are found by: the filer's and the ticker's.

    Usually one. XOM's 10-Ks were filed by Exxon Mobil Corp (34088) while the
    ticker now maps to ExxonMobil Holdings Corp (2115436); storing both lets
    a search for XOM, which looks up 2115436, find them.
    """
    ciks = {int(filing.cik)}
    if (current := ticker_cik(ticker)) is not None:
        ciks.add(current)
    return tuple(sorted(ciks))


def _recent(filed: object, today: dt.date) -> bool:
    return today - dt.date.fromisoformat(str(filed)[:10]) <= PREDECESSOR_MAX_AGE


def _no_10k_message(symbol: str, company: Company) -> str:
    filings = company.get_filings()
    forms = set(filings.to_pandas()["form"]) if len(filings) else set()
    foreign = [f for f in FOREIGN_ANNUAL_FORMS if f in forms]
    if foreign:
        return (
            f"{symbol} ({company.name}) files annual reports on Form {foreign[0]} "
            "as a foreign issuer, not 10-Ks"
        )
    return f"{symbol} ({company.name}) has no 10-K on EDGAR (an ETF, a fund or a new listing)"


def resolve_ticker(ticker: str, today: dt.date | None = None) -> TickerResolution:
    """Map ``ticker`` to the SEC registrant whose 10-Ks to read.

    Status ``unknown_ticker``: neither table nor the fund list has it.
    ``no_10k``: a company or fund that files no 10-K (TSM files 20-F, SPY
    none). The message says which, for the agent to pass on.
    """
    configure_edgar()
    symbol = ticker.strip().upper()
    # Filing dates are US Eastern days.
    today = today or dt.datetime.now(NEW_YORK).date()
    live = _live_ciks()
    live_cik = live.get(_ticker_key(symbol)) if live is not None else None
    bundled_cik = _bundled_cik(symbol)

    # The live table's CIK is current. Any other must prove it is still a
    # filer: a recent 10-K. Without the live table, the bundled CIK is all
    # there is and is trusted as before.
    candidates: list[tuple[int, bool]] = []
    if live_cik is not None:
        candidates.append((live_cik, False))
    if bundled_cik is not None and bundled_cik != live_cik:
        candidates.append((bundled_cik, live is not None))

    if not candidates:
        fund_cik = find_mutual_fund_cik(symbol)
        if fund_cik is not None:
            name = Company(fund_cik).name
            return TickerResolution(
                ticker=symbol,
                status="no_10k",
                cik=int(fund_cik),
                name=name,
                message=f"{symbol} is a mutual fund or ETF ({name}); funds file no 10-K",
            )
        return TickerResolution(
            ticker=symbol,
            status="unknown_ticker",
            message=f"{symbol} is not a ticker on SEC EDGAR",
        )

    first = None
    for cik, must_be_recent in candidates:
        company = Company(cik)
        first = first or company
        filings = company.get_filings(form="10-K", amendments=False)
        if not len(filings):
            continue
        if must_be_recent and not _recent(filings.latest(1).filing_date, today):
            continue
        predecessor = cik != candidates[0][0]
        return TickerResolution(
            ticker=symbol,
            status="ok",
            cik=cik,
            name=company.name,
            current_cik=candidates[0][0] if predecessor else None,
            message=(
                f"{symbol} is now registered as {first.name}, which has no 10-K "
                f"yet; using its predecessor {company.name}"
            )
            if predecessor
            else None,
        )
    return TickerResolution(
        ticker=symbol,
        status="no_10k",
        cik=first.cik,
        name=first.name,
        message=_no_10k_message(symbol, first),
    )


def get_latest_10k(ticker: str) -> EntityFiling:
    """Return the most recent original 10-K for ``ticker``.

    Amendments are excluded deliberately. ``get_filings`` defaults to
    ``amendments=True``, which folds 10-K/A filings into the same result set --
    AAPL has two -- so the newest entry can be an amendment rather than the
    annual report itself.

    Raises:
        FilingNotFoundError: the company or fund files no 10-K.
        UnknownTickerError: SEC has no company or fund under the ticker.
    """
    return get_latest_10ks(ticker, 1)[0]


def get_latest_10ks(ticker: str, n: int) -> list[EntityFiling]:
    """Return up to ``n`` most recent original 10-Ks for ``ticker``, newest first.

    ``latest(1)`` returns a bare ``EntityFiling`` but ``latest(n)`` for n > 1
    an ``EntityFilings``, so the result is normalised to a list. Asking for
    more years than exist returns what there is (RIVN, listed 2021, has 5).

    The ticker is resolved first (``resolve_ticker``), so XOM reads Exxon
    Mobil Corp's 10-Ks while its new holding company has none.

    Raises:
        ValueError: ``n`` is below 1 (``latest(0)`` returns None).
        FilingNotFoundError: the company or fund files no 10-K.
        UnknownTickerError: SEC has no company or fund under the ticker.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    resolution = resolve_ticker(ticker)
    if resolution.status == "unknown_ticker":
        raise UnknownTickerError(resolution.message)
    if resolution.status == "no_10k":
        raise FilingNotFoundError(resolution.message)

    filings = Company(resolution.cik).get_filings(form="10-K", amendments=False)
    latest = filings.latest(n)
    if latest is None:
        raise FilingNotFoundError(f"no 10-K filings found for ticker {ticker!r}")
    return [latest] if n == 1 else list(latest)


def filing_html(filing: EntityFiling) -> str:
    """Return the filing's primary HTML.

    ``Filing.html()`` is not reliable across companies. For inline-XBRL filings
    -- which is every modern 10-K -- it falls through to
    ``homepage.primary_html_document`` and returns None when that is absent,
    which it is for NVDA, MSFT and F among others; only AAPL happened to work.
    ``sgml().html()`` carries the same document and is populated in every case
    checked, so it is tried first and ``html()`` is kept as the fallback.

    Raises:
        UnsupportedFilingError: neither accessor produced HTML.
    """
    html = filing.sgml().html() or filing.html()
    if not html:
        raise UnsupportedFilingError(
            f"no HTML available for {filing.form} {filing.accession_no}"
        )
    return html


def _fiscal_year(filing: EntityFiling) -> int:
    """Derive the fiscal year from the period reported on, not the filing date.

    A company whose business year ends in November files the following year, so
    ``filing_date`` would label those chunks with the wrong year and a search
    filtered on ``fiscal_year`` would silently miss them.
    """
    return int(str(filing.period_of_report)[:4])


class FilingMeta(NamedTuple):
    """The facts about one filing that its chunks and archive record carry."""

    ticker: str
    accession: str
    form: str
    filing_date: str  # ISO date; a datetime.date would not serialise to JSON
    period_of_report: str
    fiscal_year: int
    # The companies the chunks are found by (``company_ciks``); set by the
    # ingest, empty where only the filing's own facts are needed.
    ciks: tuple[int, ...] = ()


def filing_meta(filing: EntityFiling, ticker: str) -> FilingMeta:
    """Describe ``filing``, upper-casing ``ticker`` so "aapl" and "AAPL" agree."""
    return FilingMeta(
        ticker=ticker.upper(),
        accession=filing.accession_no,
        form=filing.form,
        filing_date=filing.filing_date.isoformat(),
        period_of_report=str(filing.period_of_report),
        fiscal_year=_fiscal_year(filing),
    )


def archive_paths(
    ticker: str, accession: str, root: Path | None = None
) -> tuple[Path, Path]:
    """Return the ``(document, metadata)`` paths for one filing."""
    base = (root or settings.RAW_FILINGS_DIR) / ticker.upper()
    return base / f"{accession}.html.gz", base / f"{accession}.meta.json"


def archive_filing(filing: EntityFiling, ticker: str, root: Path | None = None) -> Path:
    """Archive ``filing`` as gzipped HTML plus metadata, and return its path.

    Returns early when both files already exist, so ``filing.html()`` -- the
    download -- is never called for an already archived filing. The accession
    is part of the filename, so a newly filed 10-K misses the cache and is
    picked up rather than masked by last year's copy.
    """
    accession = filing.accession_no
    document, metadata = archive_paths(ticker, accession, root)

    if document.exists() and metadata.exists():
        return document

    document.parent.mkdir(parents=True, exist_ok=True)

    payload = filing_html(filing).encode("utf-8")
    document.write_bytes(gzip.compress(payload, compresslevel=9))

    facts = filing_meta(filing, ticker)._asdict()
    del facts["ciks"]  # a search key (D-043), not a fact about the filing
    metadata.write_text(
        json.dumps(
            {
                **facts,
                # Lets the archive be pruned and later re-fetched from EDGAR
                # with proof the bytes are identical.
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes_raw": len(payload),
                "bytes_gzipped": document.stat().st_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return document


def read_archived_filing(ticker: str, accession: str, root: Path | None = None) -> str:
    """Read an archived filing back as HTML."""
    document, _ = archive_paths(ticker, accession, root)
    with gzip.open(document, "rt", encoding="utf-8") as handle:
        return handle.read()
