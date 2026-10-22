"""Search a company's 10-K filings.

With several years indexed, an unfiltered search mixes them: AAPL's supply
chain question returned FY2025, FY2024 and FY2023 chunks, two of them the same
paragraph, and "How fast is Azure growing?" ranked FY2024 first. So a search
without ``fiscal_year`` covers the latest year indexed for the company, and
the result lists the other years so a comparison can ask for them.

A company with no 10-K stored is queued for ingestion (1.3.27), and the
result says so (``queued``, ``ingesting``, ``failed``) or why it can't be
(``no_10k``, ``unknown_ticker``). ``not_indexed`` remains only for when the
queue or EDGAR is unreachable.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finai.ingestion.embed import embed_query
from finai.ingestion.fetch import ticker_cik
from finai.ingestion.jobs import request_ingestion
from finai.ingestion.parse import TARGET_ITEMS
from finai.ingestion.store import indexed_years, qdrant_client, search

MAX_K = 20

Status = Literal[
    "ok",
    "year_not_indexed",
    "queued",
    "ingesting",
    "failed",
    "no_10k",
    "unknown_ticker",
    "limit_reached",
    "not_indexed",
]


class TenKPassage(BaseModel):
    """One retrieved passage and what a citation needs."""

    model_config = ConfigDict(frozen=True)

    text: str
    ticker: str
    form: str
    fiscal_year: int
    filing_date: str
    section: str
    accession: str
    chunk_index: int
    # Cosine similarity of the passage to the question; and the reranker's
    # relevance, which set the order (None if the reranker was unavailable).
    score: float
    relevance: float | None = None


class TenKSearchResult(BaseModel):
    """Passages for one company and year, or why there are none.

    With no 10-K of the company stored, ``status`` is the ingestion request's
    (queued, ingesting, failed, no_10k, unknown_ticker, limit_reached, or
    not_indexed if it couldn't be queued) and ``message`` explains it. "year_not_indexed" means
    the requested year isn't stored; ``available_years`` says which are.
    """

    model_config = ConfigDict(frozen=True)

    status: Status
    ticker: str
    fiscal_year: int | None
    section: str | None
    available_years: list[int]
    passages: list[TenKPassage] = Field(default_factory=list)
    message: str | None = None


def query_company_10k(
    ticker: str,
    question: str,
    section: str | None = None,
    fiscal_year: int | None = None,
    k: int = 5,
    *,
    user_id: str | None = None,
) -> TenKSearchResult:
    """Find the passages of a company's 10-K most relevant to ``question``.

    ``section`` is one of "Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8".
    Without ``fiscal_year``, the latest indexed year is searched. A company
    not indexed is queued for ``user_id``, counted against their daily limit;
    without a user nothing is queued.

    Raises:
        ValueError: unknown section, or ``k`` outside 1..20.
    """
    if section is not None and section not in TARGET_ITEMS:
        raise ValueError(
            f"section must be one of {', '.join(TARGET_ITEMS)}; got {section!r}"
        )
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}; got {k}")

    symbol = ticker.strip().upper()
    client = qdrant_client()
    # By company, not ticker: GOOG finds the 10-K ingested as GOOGL (D-043).
    # A ticker SEC doesn't know has no CIK, and goes to the queue, which
    # says so.
    cik = ticker_cik(symbol)
    years = indexed_years(client, cik) if cik is not None else []
    if not years:
        if user_id is None:
            return TenKSearchResult(
                status="not_indexed",
                ticker=symbol,
                fiscal_year=fiscal_year,
                section=section,
                available_years=[],
                message=f"{symbol}'s 10-K isn't stored; fetching it needs a signed-in user.",
            )
        request = request_ingestion(symbol, user_id=user_id)
        return TenKSearchResult(
            status="not_indexed" if request.status == "unavailable" else request.status,
            ticker=symbol,
            fiscal_year=fiscal_year,
            section=section,
            available_years=[],
            message=request.message,
        )
    if fiscal_year is not None and fiscal_year not in years:
        return TenKSearchResult(
            status="year_not_indexed",
            ticker=symbol,
            fiscal_year=fiscal_year,
            section=section,
            available_years=years,
        )

    year = fiscal_year if fiscal_year is not None else years[0]
    hits = search(
        client,
        embed_query(question),
        cik,
        section=section,
        fiscal_year=year,
        k=k,
        question=question,
    )
    return TenKSearchResult(
        status="ok",
        ticker=symbol,
        fiscal_year=year,
        section=section,
        available_years=years,
        passages=[
            TenKPassage(
                text=h.chunk.text,
                # The ticker asked about, so the citation matches the
                # question even when the filing was stored under a sibling
                # share class.
                ticker=symbol,
                form=h.chunk.form,
                fiscal_year=h.chunk.fiscal_year,
                filing_date=h.chunk.filing_date,
                section=h.chunk.section,
                accession=h.chunk.accession,
                chunk_index=h.chunk.chunk_index,
                score=h.score,
                relevance=h.relevance,
            )
            for h in hits
        ],
    )
