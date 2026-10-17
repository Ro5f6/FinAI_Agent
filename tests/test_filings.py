"""Tests for the 10-K search tool.

Offline tests seed an in-memory Qdrant with two years of one company and fake
the query embedding; ``-m "qdrant and voyage"`` runs against the real index.
"""

import pytest
from qdrant_client import QdrantClient

from finai.config import settings
from finai.ingestion import embed
from finai.ingestion.chunk import chunk_filing
from finai.ingestion.fetch import FilingMeta
from finai.ingestion.jobs import IngestionRequest
from finai.ingestion.store import ensure_collection, upsert_filing
from finai.tools import filings
from finai.tools.filings import query_company_10k

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect:UserWarning"
)

_DIM = settings.EMBEDDING_DIM


def _vector(axis: int) -> list[float]:
    v = [0.0] * _DIM
    v[axis] = 1.0
    return v


@pytest.fixture
def index(monkeypatch: pytest.MonkeyPatch) -> QdrantClient:
    """AAPL FY2025 and FY2024, each with Item 1A and Item 7 chunks.

    Chunk vectors point along axis 0 (Item 1A) or 1 (Item 7); the fake query
    embedding points along axis 0, so Item 1A ranks first.
    """
    client = QdrantClient(":memory:")
    ensure_collection(client)
    for year, accession in (
        (2025, "0000320193-25-000079"),
        (2024, "0000320193-24-000123"),
    ):
        meta = FilingMeta(
            "AAPL", accession, "10-K", f"{year}-10-31", f"{year}-09-27", year
        )
        sections = {
            "Item 1A": f"Supply risk in {year}.\n\nSingle sources in {year}.",
            "Item 7": f"Net sales grew in {year}.",
        }
        payloads = chunk_filing(meta, sections, 8, 0)
        vectors = [_vector(0 if p.section == "Item 1A" else 1) for p in payloads]
        upsert_filing(client, payloads, vectors)
    monkeypatch.setattr(filings, "qdrant_client", lambda: client)
    monkeypatch.setattr(filings, "embed_query", lambda question: _vector(0))
    # Keep the embedding order; the reranker itself is tested in test_store.
    monkeypatch.setattr(
        embed,
        "rerank",
        lambda question, texts: [(i, 1.0 - i / 100) for i in range(len(texts))],
    )
    return client


def test_default_is_the_latest_indexed_year(index: QdrantClient) -> None:
    result = query_company_10k("AAPL", "supply risks")

    assert result.status == "ok"
    assert result.fiscal_year == 2025
    assert result.available_years == [2025, 2024]
    assert {p.fiscal_year for p in result.passages} == {2025}


def test_explicit_year(index: QdrantClient) -> None:
    result = query_company_10k("AAPL", "supply risks", fiscal_year=2024)

    assert result.fiscal_year == 2024
    assert {p.fiscal_year for p in result.passages} == {2024}


def test_passages_carry_citation_details(index: QdrantClient) -> None:
    [first, *_] = query_company_10k("aapl", "supply risks").passages

    assert first.ticker == "AAPL"
    assert first.form == "10-K"
    assert first.section == "Item 1A"
    assert first.accession == "0000320193-25-000079"
    assert first.filing_date == "2025-10-31"
    # Both Item 1A chunks share a vector and tie, so either may come first.
    assert first.text in ("Supply risk in 2025.", "Single sources in 2025.")
    assert first.score == pytest.approx(1.0)
    assert first.relevance == pytest.approx(1.0)


def test_section_filter(index: QdrantClient) -> None:
    result = query_company_10k("AAPL", "supply risks", section="Item 7")

    assert {p.section for p in result.passages} == {"Item 7"}


def test_k_limits_the_passages(index: QdrantClient) -> None:
    assert len(query_company_10k("AAPL", "risks", k=1).passages) == 1


def test_missing_year_says_which_years_exist(index: QdrantClient) -> None:
    result = query_company_10k("AAPL", "supply risks", fiscal_year=2019)

    assert result.status == "year_not_indexed"
    assert result.passages == []
    assert result.available_years == [2025, 2024]


def _ingestion(status: str, message: str = "msg"):
    def request(ticker: str) -> IngestionRequest:
        requested.append(ticker)
        return IngestionRequest(ticker=ticker, status=status, message=message)

    requested: list[str] = []
    request.requested = requested
    return request


@pytest.mark.parametrize(
    "status", ["queued", "ingesting", "failed", "no_10k", "unknown_ticker"]
)
def test_unindexed_company_is_queued_and_says_so(
    index: QdrantClient, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    request = _ingestion(status, "TSLA's 10-K is queued")
    monkeypatch.setattr(filings, "request_ingestion", request)

    result = query_company_10k("tsla", "supply risks")

    assert request.requested == ["TSLA"]
    assert result.status == status
    assert result.message == "TSLA's 10-K is queued"
    assert result.available_years == []
    assert result.passages == []


def test_unindexed_company_when_the_queue_is_down(
    index: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        filings, "request_ingestion", _ingestion("unavailable", "Redis down")
    )
    result = query_company_10k("TSLA", "supply risks")
    assert (result.status, result.message) == ("not_indexed", "Redis down")


def test_indexed_company_is_not_queued(
    index: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _ingestion("queued")
    monkeypatch.setattr(filings, "request_ingestion", request)
    query_company_10k("AAPL", "supply risks", fiscal_year=2019)
    query_company_10k("AAPL", "supply risks")
    assert request.requested == []


@pytest.mark.parametrize("section", ["Item 2", "Risk Factors", "1A"])
def test_unknown_section_is_rejected(index: QdrantClient, section: str) -> None:
    with pytest.raises(ValueError, match="section must be one of"):
        query_company_10k("AAPL", "risks", section=section)


@pytest.mark.parametrize("k", [0, 21])
def test_k_out_of_range_is_rejected(index: QdrantClient, k: int) -> None:
    with pytest.raises(ValueError, match="k must be between"):
        query_company_10k("AAPL", "risks", k=k)


@pytest.mark.qdrant
@pytest.mark.voyage
def test_live_latest_year_answers() -> None:
    """Needs AAPL, NVDA and MSFT FY2023-2026 ingested (1.2.19)."""
    aapl = query_company_10k("AAPL", "What are Apple's main supply chain risks?")
    assert aapl.fiscal_year == max(aapl.available_years)
    assert any("single or limited sources" in p.text for p in aapl.passages)

    msft = query_company_10k("MSFT", "How fast is Azure growing?")
    assert any(
        "Azure and other cloud services revenue increased" in p.text
        for p in msft.passages
    )

    nvda = query_company_10k(
        "NVDA",
        "How fast did data center revenue grow?",
        fiscal_year=2024,
        section="Item 7",
    )
    assert "Data Center revenue for fiscal year 2024 was up" in nvda.passages[0].text
    assert all(p.relevance is not None for p in nvda.passages)  # reranked
