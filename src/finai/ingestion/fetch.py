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
"""

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import NamedTuple

import edgar
from edgar import Company, get_identity, set_identity
from edgar.entity import EntityFiling

from finai.config import settings


class FilingNotFoundError(LookupError):
    """No filing of the requested form exists for the requested company."""


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


def get_latest_10k(ticker: str) -> EntityFiling:
    """Return the most recent original 10-K for ``ticker``.

    Amendments are excluded deliberately. ``get_filings`` defaults to
    ``amendments=True``, which folds 10-K/A filings into the same result set --
    AAPL has two -- so the newest entry can be an amendment rather than the
    annual report itself.

    Raises:
        FilingNotFoundError: the company exists but has filed no 10-K.
        CompanyNotFoundError: edgartools does not recognise the ticker.
    """
    configure_edgar()

    filings = Company(ticker).get_filings(form="10-K", amendments=False)
    latest = filings.latest()
    if latest is None:
        raise FilingNotFoundError(f"no 10-K filings found for ticker {ticker!r}")
    return latest


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

    metadata.write_text(
        json.dumps(
            {
                **filing_meta(filing, ticker)._asdict(),
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
