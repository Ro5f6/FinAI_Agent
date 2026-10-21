"""Tests for the pre-index universe: the committed S&P 500 file and the
resolution report. Offline; the live run's results are in docs/CONTEXT.md."""

from pathlib import Path

import httpx
import pytest

from finai.ingestion import universe
from finai.ingestion.fetch import TickerResolution
from finai.ingestion.universe import (
    Checked,
    Constituent,
    check_universe,
    read_tickers,
    read_universe,
    report,
    shared_ciks,
)

SP500 = Path(__file__).parents[1] / "reference" / "sp500-2026-10-20.csv"


def _ok(ticker: str, cik: int, current: int | None = None) -> TickerResolution:
    return TickerResolution(
        ticker=ticker,
        status="ok",
        cik=cik,
        name=f"{ticker} Inc.",
        current_cik=current,
        message="using its predecessor" if current else None,
    )


# --- the committed file -------------------------------------------------------


def test_sp500_file_has_503_share_classes_of_500_companies() -> None:
    rows = read_universe(SP500)
    tickers = [r.ticker for r in rows]
    assert len(rows) == 503
    assert len(set(tickers)) == 503
    assert len({r.cik for r in rows}) == 500  # GOOGL/GOOG, FOXA/FOX, NWSA/NWS
    assert {"AAPL", "BRK.B", "BF.B", "GOOG", "GOOGL"} <= set(tickers)
    assert all(r.cik for r in rows)


def test_sp500_file_records_its_sources() -> None:
    header = [line for line in SP500.read_text().splitlines() if line.startswith("#")]
    text = "\n".join(header)
    assert "As of 20-Oct-2026" in text
    assert "ssga.com" in text
    assert "wikipedia.org" in text
    assert "retrieved 2026-10-21" in text


# --- reading ----------------------------------------------------------------------


def test_read_skips_source_lines_and_normalises(tmp_path: Path) -> None:
    path = tmp_path / "u.csv"
    path.write_text(
        "# source: test\n# more\nticker,name,cik\n brk.b ,Berkshire,1067983\nPLTR,,\n"
    )
    assert read_universe(path) == [
        Constituent("BRK.B", "Berkshire", 1067983),
        Constituent("PLTR", None, None),
    ]
    assert read_tickers(path) == ["BRK.B", "PLTR"]


def test_ticker_column_alone_is_enough(tmp_path: Path) -> None:
    path = tmp_path / "u.csv"
    path.write_text("ticker\nAAPL\nMSFT\n")
    assert read_tickers(path) == ["AAPL", "MSFT"]


# --- checking ---------------------------------------------------------------------

_RESOLUTIONS = {
    "AAPL": _ok("AAPL", 320193),
    "GOOGL": _ok("GOOGL", 1652044),
    "GOOG": _ok("GOOG", 1652044),
    "XOM": _ok("XOM", 34088, current=2115436),
    "HONA": TickerResolution(
        ticker="HONA", status="no_10k", cik=2089271, message="HONA has no 10-K on EDGAR"
    ),
    "WRONG": _ok("WRONG", 999),
}


def _check(constituents: list[Constituent]) -> list[Checked]:
    def resolve(ticker: str) -> TickerResolution:
        if ticker == "DOWN":
            raise httpx.ConnectError("SEC unreachable")
        return _RESOLUTIONS[ticker]

    return check_universe(constituents, resolve=resolve)


def test_predecessor_cik_listed_by_the_source_is_not_a_mismatch() -> None:
    # The source may list XOM under either the new registrant or the old one.
    for listed in (34088, 2115436):
        [c] = _check([Constituent("XOM", "Exxon", listed)])
        assert c.ingestible
        assert not c.cik_mismatch


def test_a_different_company_is_a_mismatch() -> None:
    [c] = _check([Constituent("WRONG", None, 123)])
    assert c.cik_mismatch


def test_network_failure_is_recorded_per_ticker() -> None:
    checked = _check(
        [Constituent("DOWN", None, None), Constituent("AAPL", None, 320193)]
    )
    assert checked[0].error == "ConnectError: SEC unreachable"
    assert not checked[0].ingestible
    assert checked[1].ingestible


def test_shared_ciks_are_share_classes() -> None:
    checked = _check([Constituent(t, None, None) for t in ("AAPL", "GOOGL", "GOOG")])
    assert shared_ciks(checked) == {1652044: ["GOOGL", "GOOG"]}


def test_report_lists_every_kind_of_problem() -> None:
    checked = _check(
        [
            Constituent("AAPL", None, 320193),
            Constituent("GOOGL", None, 1652044),
            Constituent("GOOG", None, 1652044),
            Constituent("XOM", None, 2115436),
            Constituent("HONA", None, 2089271),
            Constituent("WRONG", None, 123),
            Constituent("DOWN", None, None),
        ]
    )
    text = report(checked)
    assert text.startswith("7 tickers: 5 resolve to a 10-K filer, 2 don't.")
    assert "HONA: no_10k - HONA has no 10-K on EDGAR" in text
    assert "DOWN: ConnectError: SEC unreachable" in text
    assert "XOM: using its predecessor" in text
    assert "WRONG: file 123, resolved 999" in text
    assert "CIK 1652044: GOOGL, GOOG" in text


def test_cli_exit_code_says_whether_everything_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(universe, "resolve_ticker", lambda t: _RESOLUTIONS[t])
    good = tmp_path / "good.csv"
    good.write_text("ticker\nAAPL\nXOM\n")
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker\nAAPL\nHONA\n")

    assert universe.main([str(good)]) == 0
    assert universe.main([str(bad)]) == 1
    assert "HONA: no_10k" in capsys.readouterr().out
