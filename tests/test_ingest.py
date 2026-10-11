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

    def __init__(self, year: int = 2025) -> None:
        self.accession_no = f"0000320193-{year % 100}-000079"
        self.filing_date = dt.date(year, 10, 31)
        self.period_of_report = f"{year}-09-27"


def _filings(ticker: str, n: int) -> list[_Filing]:
    if ticker.upper() == "NOPE":
        raise FilingNotFoundError(f"no 10-K filings found for ticker {ticker!r}")
    available = 2 if ticker.upper() == "RIVN" else 5
    return [_Filing(2025 - i) for i in range(min(n, available))]


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> QdrantClient:
    client = QdrantClient(":memory:")
    monkeypatch.setattr(ingest, "get_latest_10ks", _filings)
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
    [result] = ingest.ingest_ticker("aapl")

    assert result.ticker == "AAPL"
    assert result.fiscal_year == 2025
    assert result.sections == ("Item 1A",)
    assert result.chunks > 1
    assert pipeline.count(settings.COLLECTION_NAME).count == result.chunks


def test_cross_reference_items_are_reported(pipeline: QdrantClient) -> None:
    [result] = ingest.ingest_ticker("AAPL")

    assert set(result.vestigial) == {"Item 8"}
    assert "dropped Item 8" in result.summary()


def test_rerun_leaves_the_count_unchanged(pipeline: QdrantClient) -> None:
    [first] = ingest.ingest_ticker("AAPL")
    [second] = ingest.ingest_ticker("AAPL")

    assert second.chunks == first.chunks
    assert second.stale_removed == 0
    assert pipeline.count(settings.COLLECTION_NAME).count == first.chunks


def test_main_prints_a_summary(pipeline, capsys: pytest.CaptureFixture) -> None:
    assert ingest.main(["--ticker", "AAPL"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("AAPL FY2025 0000320193-25-000079:")


def test_unknown_company_exits_1_with_a_message(
    pipeline, capsys: pytest.CaptureFixture
) -> None:
    assert ingest.main(["--ticker", "NOPE"]) == 1
    assert "no 10-K filings found for ticker 'NOPE'" in capsys.readouterr().err


def test_years_are_stored_separately(pipeline: QdrantClient) -> None:
    results = ingest.ingest_ticker("AAPL", years=3)

    assert [r.fiscal_year for r in results] == [2025, 2024, 2023]
    assert len({r.accession for r in results}) == 3
    total = sum(r.chunks for r in results)
    assert pipeline.count(settings.COLLECTION_NAME).count == total


def test_several_tickers_and_a_failure_in_between(
    pipeline: QdrantClient, capsys: pytest.CaptureFixture
) -> None:
    """One bad ticker doesn't stop the others; the exit code reports it."""
    assert ingest.main(["--ticker", "AAPL", "NOPE", "RIVN", "--years", "2"]) == 1

    out = capsys.readouterr()
    assert out.out.count(" FY") == 4  # AAPL 2 + RIVN 2
    assert "NOPE" in out.err


def test_fewer_years_than_asked_is_reported(
    pipeline: QdrantClient, capsys: pytest.CaptureFixture
) -> None:
    assert ingest.main(["--ticker", "RIVN", "--years", "4"]) == 0

    assert "RIVN: only 2 of 4 years exist" in capsys.readouterr().out


def test_years_below_one_is_rejected() -> None:
    with pytest.raises(SystemExit):
        ingest.main(["--ticker", "AAPL", "--years", "0"])


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
