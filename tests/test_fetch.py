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

import httpx
import pandas as pd
import pytest

from finai.config import settings
from finai.ingestion import fetch
from finai.ingestion.fetch import (
    FilingNotFoundError,
    TickerResolution,
    UnknownTickerError,
    UnsupportedFilingError,
    archive_filing,
    archive_paths,
    configure_edgar_identity,
    filing_html,
    filing_meta,
    get_latest_10k,
    get_latest_10ks,
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
    """Mimics edgartools: ``latest(1)`` returns one filing, ``latest(n)`` a
    collection of up to n, and None when there are no filings."""

    def __init__(self, latest: object | None, available: int = 5) -> None:
        self._latest = latest
        self._available = available

    def latest(self, n: int = 1) -> object | None:
        if self._latest is None or n < 1:
            return None
        if n == 1:
            return self._latest
        return [self._latest] * min(n, self._available)


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
    monkeypatch.setattr(
        fetch,
        "resolve_ticker",
        lambda ticker: TickerResolution(ticker=ticker, status="ok", cik=320193),
    )
    return captured


def test_get_latest_10k_excludes_amendments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 10-K/A must never stand in for the annual report itself."""
    sentinel = object()
    captured = _stub_company(monkeypatch, sentinel)

    assert get_latest_10k("AAPL") is sentinel
    assert captured["ticker"] == 320193  # by the resolved CIK, not the ticker
    assert captured["form"] == "10-K"
    assert captured["amendments"] is False


def test_get_latest_10ks_always_returns_a_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """latest(1) returns a bare filing; callers still get a list."""
    sentinel = object()
    _stub_company(monkeypatch, sentinel)

    assert get_latest_10ks("AAPL", 1) == [sentinel]
    assert get_latest_10ks("AAPL", 3) == [sentinel] * 3


def test_get_latest_10ks_returns_what_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    _stub_company(monkeypatch, sentinel)

    assert len(get_latest_10ks("AAPL", 10)) == 5


def test_get_latest_10ks_rejects_zero_years() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        get_latest_10ks("AAPL", 0)


def test_get_latest_10k_raises_when_no_filings(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty result set returns None from latest() rather than raising."""
    _stub_company(monkeypatch, None)

    with pytest.raises(FilingNotFoundError, match="NOFILINGS"):
        get_latest_10k("NOFILINGS")


def test_get_latest_10ks_raises_for_an_unknown_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fetch,
        "resolve_ticker",
        lambda t: TickerResolution(
            ticker="ZZZZQ", status="unknown_ticker", message="nope"
        ),
    )
    with pytest.raises(UnknownTickerError, match="nope"):
        get_latest_10ks("ZZZZQ", 1)


def test_get_latest_10ks_raises_with_the_reason_when_no_10k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fetch,
        "resolve_ticker",
        lambda t: TickerResolution(ticker="TSM", status="no_10k", message="files 20-F"),
    )
    with pytest.raises(FilingNotFoundError, match="20-F"):
        get_latest_10ks("TSM", 1)


# --- ticker resolution (1.3.26) ------------------------------------------------

TODAY = dt.date(2026, 10, 17)


class _FakeFiling:
    def __init__(self, filed: str) -> None:
        self.filing_date = dt.date.fromisoformat(filed)


class _FakeFilings:
    def __init__(self, forms: list[tuple[str, str]]) -> None:
        self._forms = forms  # (form, filed), newest first

    def __len__(self) -> int:
        return len(self._forms)

    def latest(self, n: int = 1) -> _FakeFiling:
        return _FakeFiling(self._forms[0][1])

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({"form": [f for f, _ in self._forms]})


# CIK -> (name, filings), modelled on what EDGAR returned on 2026-10-17.
_EDGAR = {
    320193: ("Apple Inc.", [("10-K", "2025-10-31"), ("10-Q", "2026-08-01")]),
    2115436: ("ExxonMobil Holdings Corp", [("8-K", "2026-10-16")]),
    34088: ("EXXON MOBIL CORP", [("10-K", "2026-02-18"), ("10-Q", "2026-08-03")]),
    1046179: ("TAIWAN SEMICONDUCTOR", [("6-K", "2026-10-01"), ("20-F", "2026-04-15")]),
    884394: ("SPDR S&P 500 ETF TRUST", [("N-30B-2", "2026-06-01")]),
    1174610: ("PROSHARES TRUST", []),
    # A ticker reused by a new company; the old holder stopped filing in 2019.
    900001: ("New Co Inc.", [("S-1", "2026-09-01")]),
    900002: ("Old Co Corp", [("10-K", "2019-03-01")]),
    # Delisted: only in the bundled table, last 10-K in 2021.
    900003: ("Gone Corp", [("10-K", "2021-03-01")]),
}


class _FakeCompany:
    def __init__(self, cik: int) -> None:
        self.cik = cik
        self.name, self._filings = _EDGAR[cik]

    def get_filings(
        self, form: str | None = None, amendments: bool = True
    ) -> _FakeFilings:
        rows = [r for r in self._filings if form is None or r[0] == form]
        return _FakeFilings(rows)


@pytest.fixture
def edgar_tables(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    """Fake live and bundled ticker tables over the fake EDGAR above."""
    tables = {
        "live": {"AAPL": 320193, "BRK-B": 320193, "XOM": 2115436, "TSM": 1046179,
                 "SPY": 884394, "OLD": 900001},
        "bundled": {"AAPL": 320193, "XOM": 34088, "TSM": 1046179, "SPY": 884394,
                    "OLD": 900002, "GONE": 900003},
        "funds": {"FB": 1174610},
    }  # fmt: skip
    monkeypatch.setattr(fetch, "configure_edgar", lambda: None)
    monkeypatch.setattr(fetch, "Company", _FakeCompany)
    monkeypatch.setattr(fetch, "_live_ciks", lambda: tables["live"])
    monkeypatch.setattr(
        fetch, "_bundled_cik", lambda s: tables["bundled"].get(s.replace(".", "-"))
    )
    monkeypatch.setattr(fetch, "find_mutual_fund_cik", lambda s: tables["funds"].get(s))
    return tables


def _resolve(ticker: str) -> TickerResolution:
    return fetch.resolve_ticker(ticker, today=TODAY)


def test_resolves_a_10k_filer(edgar_tables) -> None:
    r = _resolve(" aapl ")
    assert (r.ticker, r.status, r.cik, r.current_cik) == ("AAPL", "ok", 320193, None)


def test_share_class_dot_matches_secs_dash(edgar_tables) -> None:
    assert _resolve("BRK.B").cik == 320193


def test_reorganised_company_uses_its_recent_predecessor(edgar_tables) -> None:
    r = _resolve("XOM")
    assert (r.status, r.cik, r.current_cik) == ("ok", 34088, 2115436)
    assert "ExxonMobil Holdings Corp" in r.message


def test_reused_ticker_does_not_fall_back_to_a_dead_company(edgar_tables) -> None:
    r = _resolve("OLD")
    assert (r.status, r.cik) == ("no_10k", 900001)


def test_delisted_ticker_only_in_bundled_table_needs_a_recent_10k(edgar_tables) -> None:
    assert _resolve("GONE").status == "no_10k"


def test_without_the_live_table_the_bundled_cik_is_trusted(
    edgar_tables, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "_live_ciks", lambda: None)
    assert (_resolve("GONE").status, _resolve("XOM").cik) == ("ok", 34088)


def test_foreign_issuer_names_its_annual_form(edgar_tables) -> None:
    r = _resolve("TSM")
    assert r.status == "no_10k"
    assert "Form 20-F" in r.message


def test_etf_has_no_10k(edgar_tables) -> None:
    r = _resolve("SPY")
    assert r.status == "no_10k"
    assert "no 10-K on EDGAR" in r.message


def test_fund_ticker_is_reported_as_a_fund(edgar_tables) -> None:
    # FB is no longer Meta: SEC lists it only as a ProShares fund ticker.
    r = _resolve("FB")
    assert r.status == "no_10k"
    assert "mutual fund or ETF (PROSHARES TRUST)" in r.message


def test_unknown_ticker(edgar_tables) -> None:
    r = _resolve("ZZZZQ")
    assert (r.status, r.cik) == ("unknown_ticker", None)


def test_live_table_is_cached_for_a_day(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads = []
    clock = [1000.0]

    def download(url: str) -> dict:
        downloads.append(url)
        return {"0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."}}

    monkeypatch.setattr(fetch, "download_json", download)
    monkeypatch.setattr(fetch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(fetch, "_sec_tickers", None)
    assert fetch._live_ciks() == {"AAPL": 320193}
    fetch._live_ciks()
    assert len(downloads) == 1
    clock[0] += fetch.SEC_TICKERS_TTL_S + 1
    fetch._live_ciks()
    assert len(downloads) == 2


def test_live_table_failure_keeps_the_last_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(url: str) -> dict:
        raise httpx.ConnectError("SEC down")

    monkeypatch.setattr(fetch, "download_json", fail)
    monkeypatch.setattr(fetch, "_sec_tickers", None)
    assert fetch._live_ciks() is None
    monkeypatch.setattr(fetch, "_sec_tickers", (-1e12, {"AAPL": 320193}))
    assert fetch._live_ciks() == {"AAPL": 320193}


@pytest.mark.network
@pytest.mark.parametrize(
    ("ticker", "status", "cik"),
    [
        ("AAPL", "ok", 320193),
        ("BRK.B", "ok", 1067983),
        ("XOM", "ok", 34088),  # predecessor of ExxonMobil Holdings Corp
        ("TSM", "no_10k", 1046179),
        ("SPY", "no_10k", 884394),
        ("ZZZZQ", "unknown_ticker", None),
    ],
)
def test_resolve_against_live_edgar(ticker: str, status: str, cik: int | None) -> None:
    r = fetch.resolve_ticker(ticker)
    assert (r.status, r.cik) == (status, cik)


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
