"""Tests for EDGAR access.

The offline tests stub out ``Company`` so the default suite neither touches the
network nor consumes EDGAR rate limit. The live call is marked ``network`` and
deselected unless asked for.
"""

import datetime as dt
import gzip
import json
import os
from pathlib import Path
from typing import Any

import pytest

from finai.config import settings
from finai.ingestion import fetch
from finai.ingestion.fetch import (
    FilingNotFoundError,
    UnsupportedFilingError,
    archive_filing,
    archive_paths,
    configure_edgar_identity,
    filing_html,
    filing_meta,
    get_latest_10k,
    read_archived_filing,
)


def test_configure_registers_identity_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)

    registered = configure_edgar_identity()

    # edgartools communicates the identity to the SEC through this variable.
    assert os.environ["EDGAR_IDENTITY"] == settings.SEC_IDENTITY
    assert registered == settings.SEC_IDENTITY


def test_configure_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetching several filings re-registers the identity; that must be safe."""
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)

    assert configure_edgar_identity() == configure_edgar_identity()


def test_identity_is_not_anonymous() -> None:
    """A blank identity is what the SEC rejects, so guard against it."""
    assert configure_edgar_identity().strip() != ""


class _StubFilings:
    def __init__(self, latest: object | None) -> None:
        self._latest = latest

    def latest(self) -> object | None:
        return self._latest


def _stub_company(
    monkeypatch: pytest.MonkeyPatch, latest: object | None
) -> dict[str, Any]:
    """Replace Company with a stub and capture the get_filings keyword arguments."""
    captured: dict[str, Any] = {}

    class _StubCompany:
        def __init__(self, ticker: str) -> None:
            captured["ticker"] = ticker

        def get_filings(self, **kwargs: Any) -> _StubFilings:
            captured.update(kwargs)
            return _StubFilings(latest)

    monkeypatch.setattr(fetch, "Company", _StubCompany)
    return captured


def test_get_latest_10k_excludes_amendments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 10-K/A must never stand in for the annual report itself."""
    sentinel = object()
    captured = _stub_company(monkeypatch, sentinel)

    assert get_latest_10k("AAPL") is sentinel
    assert captured["ticker"] == "AAPL"
    assert captured["form"] == "10-K"
    assert captured["amendments"] is False


def test_get_latest_10k_raises_when_no_filings(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty result set returns None from latest() rather than raising."""
    _stub_company(monkeypatch, None)

    with pytest.raises(FilingNotFoundError, match="NOFILINGS"):
        get_latest_10k("NOFILINGS")


@pytest.mark.network
def test_get_latest_10k_against_live_edgar() -> None:
    filing = get_latest_10k("AAPL")

    assert filing.form == "10-K"
    assert filing.accession_no
    assert filing.filing_date is not None


class _StubSgml:
    """Stand-in for the SGML wrapper, whose html() is the preferred accessor."""

    def __init__(self, fetch_html: Any) -> None:
        self._fetch_html = fetch_html

    def html(self) -> str | None:
        return self._fetch_html()


class _StubFiling:
    """Minimal stand-in for EntityFiling, counting document downloads."""

    form = "10-K"
    accession_no = "0000320193-25-000079"
    filing_date = dt.date(2026, 2, 15)
    # Deliberately a different year from filing_date: a November year end
    # filed the following February is exactly where the two diverge.
    period_of_report = "2025-11-30"

    def __init__(
        self,
        html: str = "<html><body>Item 1A. Risk Factors</body></html>",
        sgml_html: str | None = None,
        use_sgml: bool = True,
    ) -> None:
        self._html = html
        # None models the real failure: Filing.html() returns None for most
        # inline-XBRL filings while sgml().html() still holds the document.
        self._sgml_html = (
            sgml_html if sgml_html is not None else (html if use_sgml else None)
        )
        self.html_calls = 0

    def _download(self, body: str | None) -> str | None:
        if body:
            self.html_calls += 1
        return body

    def sgml(self) -> _StubSgml:
        return _StubSgml(lambda: self._download(self._sgml_html))

    def html(self) -> str | None:
        return self._download(self._html)


def test_archive_writes_gzipped_html_and_metadata(tmp_path: Path) -> None:
    filing = _StubFiling()

    document = archive_filing(filing, "aapl", root=tmp_path)

    assert document.suffixes[-2:] == [".html", ".gz"]
    # Ticker is upper-cased so "aapl" and "AAPL" cannot create two directories.
    assert document.parent.name == "AAPL"
    with gzip.open(document, "rt", encoding="utf-8") as handle:
        assert "Risk Factors" in handle.read()


def test_archive_metadata_contents(tmp_path: Path) -> None:
    filing = _StubFiling()

    archive_filing(filing, "AAPL", root=tmp_path)
    _, metadata = archive_paths("AAPL", filing.accession_no, root=tmp_path)
    meta = json.loads(metadata.read_text())

    assert meta["accession"] == filing.accession_no
    assert meta["form"] == "10-K"
    # A date object would raise TypeError from json.dump, so it must be a string.
    assert meta["filing_date"] == "2026-02-15"
    assert isinstance(meta["sha256"], str) and len(meta["sha256"]) == 64
    # Sizes are recorded for both; gzip only beats raw on realistic input,
    # which test_archive_round_trips_through_gzip covers.
    assert meta["bytes_raw"] > 0
    assert meta["bytes_gzipped"] > 0


def test_fiscal_year_comes_from_period_not_filing_date(tmp_path: Path) -> None:
    """The regression guard for mislabelled chunks on non-calendar year ends."""
    filing = _StubFiling()

    archive_filing(filing, "AAPL", root=tmp_path)
    _, metadata = archive_paths("AAPL", filing.accession_no, root=tmp_path)

    assert json.loads(metadata.read_text())["fiscal_year"] == 2025  # not 2026


def test_filing_meta_describes_the_filing() -> None:
    meta = filing_meta(_StubFiling(), "aapl")

    assert meta.ticker == "AAPL"
    assert meta.accession == "0000320193-25-000079"
    assert meta.form == "10-K"
    assert meta.filing_date == "2026-02-15"
    assert meta.period_of_report == "2025-11-30"
    assert meta.fiscal_year == 2025


def test_archive_metadata_matches_filing_meta(tmp_path: Path) -> None:
    """Chunks and the archive describe a filing from the same source."""
    filing = _StubFiling()

    archive_filing(filing, "AAPL", root=tmp_path)
    _, metadata = archive_paths("AAPL", filing.accession_no, root=tmp_path)
    archived = json.loads(metadata.read_text())

    assert filing_meta(filing, "AAPL")._asdict().items() <= archived.items()


def test_archive_skips_download_when_already_cached(tmp_path: Path) -> None:
    """The whole point: an archived filing must not be downloaded again."""
    filing = _StubFiling()

    archive_filing(filing, "AAPL", root=tmp_path)
    assert filing.html_calls == 1

    archive_filing(filing, "AAPL", root=tmp_path)
    assert filing.html_calls == 1  # unchanged: no second download


def test_archive_is_not_reused_for_a_different_accession(tmp_path: Path) -> None:
    """A newly filed 10-K must miss the cache rather than read last year's."""
    old = _StubFiling("<html>old</html>")
    archive_filing(old, "AAPL", root=tmp_path)

    new = _StubFiling("<html>new</html>")
    new.accession_no = "0000320193-26-000001"
    archive_filing(new, "AAPL", root=tmp_path)

    assert new.html_calls == 1
    assert "new" in read_archived_filing("AAPL", new.accession_no, root=tmp_path)
    assert "old" in read_archived_filing("AAPL", old.accession_no, root=tmp_path)


def test_archive_round_trips_through_gzip(tmp_path: Path) -> None:
    filing = _StubFiling("<html><body>" + "supply chain risk " * 500 + "</body></html>")

    archive_filing(filing, "AAPL", root=tmp_path)

    assert (
        read_archived_filing("AAPL", filing.accession_no, root=tmp_path)
        == filing.html()
    )


def test_settings_defaults_separate_archive_from_cache() -> None:
    """The archive and the library cache must not share a directory."""
    assert settings.RAW_FILINGS_DIR != settings.EDGAR_LOCAL_DATA_DIR
    assert 1 <= settings.EDGAR_RATE_LIMIT_PER_SEC <= 10


def test_filing_html_prefers_sgml() -> None:
    """sgml().html() is populated where Filing.html() is not."""
    filing = _StubFiling(
        html="<html>from html()</html>", sgml_html="<html>from sgml</html>"
    )

    assert "from sgml" in filing_html(filing)


def test_filing_html_falls_back_when_sgml_is_empty() -> None:
    filing = _StubFiling(html="<html>fallback</html>", use_sgml=False)

    assert "fallback" in filing_html(filing)


def test_filing_html_raises_when_no_document_is_available() -> None:
    """The real NVDA/MSFT failure. Fail clearly here rather than with
    AttributeError on None several frames later."""
    filing = _StubFiling(html="", sgml_html=None)

    with pytest.raises(UnsupportedFilingError, match="no HTML available"):
        filing_html(filing)
