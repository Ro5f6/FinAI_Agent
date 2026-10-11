"""Tests for the ingest CLI.

Offline tests replace the SEC, Voyage and Qdrant with stand-ins and run the
real extract, clean, chunk and upsert code in between.
"""

import datetime as dt
import subprocess
import sys

import pytest
from qdrant_client import QdrantClient

from finai.config import settings
from finai.ingestion import ingest
from finai.ingestion.fetch import FilingNotFoundError

# The offline pipeline uses in-memory Qdrant, which ignores payload indexes
# and says so; the server tests (-m qdrant) cover the indexes.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect:UserWarning"
)

_RISKS = "Item 1A. Risk Factors\n\n" + " ".join(
    f"Risk sentence {i} about suppliers." for i in range(400)
)
_ITEM_8 = "Item 8. Financial Statements\n\nSee page 60."


class _Filing:
    """Stand-in for EntityFiling: the attributes ``filing_meta`` reads."""

    form = "10-K"
    accession_no = "0000320193-25-000079"
    filing_date = dt.date(2025, 10, 31)
    period_of_report = "2025-09-27"


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> QdrantClient:
    client = QdrantClient(":memory:")
    monkeypatch.setattr(ingest, "get_latest_10k", lambda ticker: _Filing())
    monkeypatch.setattr(ingest, "archive_filing", lambda filing, ticker: None)
    monkeypatch.setattr(
        ingest,
        "extract_filing_items",
        lambda filing: {"Item 1A": _RISKS, "Item 8": _ITEM_8},
    )
    monkeypatch.setattr(
        ingest,
        "embed_documents",
        lambda texts: [
            [1.0, float(i)] + [0.0] * (settings.EMBEDDING_DIM - 2)
            for i in range(len(texts))
        ],
    )
    monkeypatch.setattr(ingest, "qdrant_client", lambda: client)
    return client


def test_ingest_stores_every_chunk(pipeline: QdrantClient) -> None:
    result = ingest.ingest_ticker("aapl")

    assert result.ticker == "AAPL"
    assert result.fiscal_year == 2025
    assert result.sections == ("Item 1A",)
    assert result.chunks > 1
    assert pipeline.count(settings.COLLECTION_NAME).count == result.chunks


def test_cross_reference_items_are_reported(pipeline: QdrantClient) -> None:
    result = ingest.ingest_ticker("AAPL")

    assert set(result.vestigial) == {"Item 8"}
    assert "dropped Item 8" in result.summary()


def test_rerun_leaves_the_count_unchanged(pipeline: QdrantClient) -> None:
    first = ingest.ingest_ticker("AAPL")
    second = ingest.ingest_ticker("AAPL")

    assert second.chunks == first.chunks
    assert second.stale_removed == 0
    assert pipeline.count(settings.COLLECTION_NAME).count == first.chunks


def test_main_prints_a_summary(pipeline, capsys: pytest.CaptureFixture) -> None:
    assert ingest.main(["--ticker", "AAPL"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("AAPL FY2025 0000320193-25-000079:")


def test_unknown_company_exits_1_with_a_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def no_filing(ticker: str):
        raise FilingNotFoundError(f"no 10-K filings found for ticker {ticker!r}")

    monkeypatch.setattr(ingest, "get_latest_10k", no_filing)

    assert ingest.main(["--ticker", "NOPE"]) == 1
    assert "no 10-K filings found for ticker 'NOPE'" in capsys.readouterr().err


def test_ticker_is_required() -> None:
    with pytest.raises(SystemExit):
        ingest.main([])


@pytest.mark.network
@pytest.mark.voyage
@pytest.mark.qdrant
def test_cli_ingests_aapl_end_to_end() -> None:
    """The command as a user runs it, against SEC, Voyage and the compose Qdrant."""
    run = subprocess.run(
        [sys.executable, "-m", "finai.ingestion.ingest", "--ticker", "AAPL"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith("AAPL FY")
    assert "0 stale removed" in run.stdout
