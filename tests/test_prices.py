"""Tests for price history.

Offline tests use a small hand-built series whose figures are worked out by
hand in the comments, and replay Alpaca's response shapes (recorded
2026-10-15) through httpx's mock transport. ``-m alpaca`` calls the API.
"""

import datetime as dt
import math
from itertools import pairwise

import httpx
import pytest

from finai.tools import prices
from finai.tools.market_data import TickerNotFoundError
from finai.tools.prices import (
    NO_10K,
    Bar,
    PriceHistoryError,
    compute_price_history,
    get_price_history,
)

D = dt.date


def _bar(date: dt.date, close: float, high: float | None = None, low=None) -> Bar:
    return Bar(date, high if high is not None else close, low or close, close)


# Dividend-adjusted closes. The last bar is Monday 2026-06-15.
ADJUSTED = [
    _bar(D(2021, 6, 10), 50),  # 5y base: last close on or before 2021-06-15
    _bar(D(2023, 6, 15), 80),  # 3y base: the start date itself
    _bar(D(2025, 6, 13), 100),  # 1y base: 2025-06-15 is a Sunday
    _bar(D(2025, 9, 1), 120),  # peak
    _bar(D(2025, 12, 1), 90),  # trough: 90 / 120 - 1 = -25%
    _bar(D(2026, 3, 2), 110),
    _bar(D(2026, 6, 15), 125),
]
# Split-only bars: higher than the adjusted closes, as before a dividend.
SPLIT_ONLY = [
    _bar(D(2025, 6, 13), 101, high=200, low=10),  # a year and two days back: out
    _bar(D(2025, 9, 1), 121, high=130, low=118),
    _bar(D(2025, 12, 1), 91, high=95, low=85),
    _bar(D(2026, 6, 15), 126, high=127, low=124),
]
# The first pair starts before the first bar, the last ends after the last
# bar, so only fiscal 2025 (2023-06-30 to 2025-06-30) is complete.
YEAR_ENDS = [D(2020, 12, 31), D(2023, 6, 30), D(2025, 6, 30), D(2026, 12, 31)]


def _history(period="1y", adjusted=ADJUSTED, ends=YEAR_ENDS, note=None):
    return compute_price_history("TEST", period, adjusted, SPLIT_ONLY, ends, note)


def test_one_year_return_from_last_close_before_start() -> None:
    h = _history("1y")
    assert h.requested_start == D(2025, 6, 15)
    assert h.start_date == D(2025, 6, 13)
    assert h.end_date == D(2026, 6, 15)
    assert h.full_period
    assert h.total_return == pytest.approx(0.25)  # 125 / 100 - 1
    assert h.annualised_return is None


def test_one_year_volatility() -> None:
    # Log returns ln(120/100), ln(90/120), ln(110/90), ln(125/110); sample
    # standard deviation 0.231054..., times sqrt(252).
    assert _history("1y").annualised_volatility == pytest.approx(3.6679385030817)


def test_max_drawdown_with_dates() -> None:
    h = _history("1y")
    assert h.max_drawdown == pytest.approx(-0.25)
    assert (h.max_drawdown_peak, h.max_drawdown_trough) == (
        D(2025, 9, 1),
        D(2025, 12, 1),
    )


def test_three_and_five_years_annualised() -> None:
    three = _history("3y")
    assert three.start_date == D(2023, 6, 15)
    assert three.total_return == pytest.approx(0.5625)  # 125 / 80 - 1
    # 1,096 days: 1.5625 ** (365.25 / 1096) - 1
    assert three.annualised_return == pytest.approx(0.160357833287)

    five = _history("5y")
    assert five.start_date == D(2021, 6, 10)
    assert five.total_return == pytest.approx(1.5)  # 125 / 50 - 1
    # 1,831 days: 2.5 ** (365.25 / 1831) - 1
    assert five.annualised_return == pytest.approx(0.200553542810)


def test_short_listing_starts_at_first_bar_and_says_so() -> None:
    h = _history("5y", adjusted=ADJUSTED[1:])
    assert not h.full_period
    assert h.requested_start == D(2021, 6, 15)
    assert h.start_date == D(2023, 6, 15)
    assert h.total_return == pytest.approx(0.5625)
    # Annualised over the 1,096 days there are, not five years.
    assert h.annualised_return == pytest.approx(0.160357833287)


def test_52_week_range_from_split_only_bars_within_the_year() -> None:
    h = _history("5y")
    assert (h.week_52_high, h.week_52_high_date) == (130, D(2025, 9, 1))
    assert (h.week_52_low, h.week_52_low_date) == (85, D(2025, 12, 1))


def test_fiscal_year_returns_only_for_complete_years() -> None:
    h = _history("1y")
    [fy] = h.fiscal_year_returns
    assert fy.fiscal_year == 2025
    assert (fy.start_date, fy.end_date) == (D(2023, 6, 30), D(2025, 6, 30))
    # Last closes on or before each end: 80 (2023-06-15), 100 (2025-06-13).
    assert fy.total_return == pytest.approx(0.25)
    assert h.fiscal_years_note is None


def test_fiscal_year_notes() -> None:
    assert _history(ends=[], note=NO_10K).fiscal_years_note == NO_10K
    assert (
        "no complete fiscal year" in _history(ends=[D(2026, 12, 31)]).fiscal_years_note
    )


def test_fiscal_year_returns_newest_first() -> None:
    ends = [D(2023, 6, 30), D(2025, 8, 31), D(2026, 1, 31)]
    years = [fy.fiscal_year for fy in _history(ends=ends).fiscal_year_returns]
    assert years == [2026, 2025]


def test_invalid_period() -> None:
    with pytest.raises(ValueError, match="period"):
        _history("2y")
    with pytest.raises(ValueError, match="period"):
        get_price_history("AAPL", "10y")


def test_bar_date_is_new_york_trading_day() -> None:
    assert prices._bar_date("2021-11-10T05:00:00Z") == D(2021, 11, 10)
    assert prices._bar_date("2024-06-10T04:00:00Z") == D(2024, 6, 10)


# --- HTTP: recorded response shapes -----------------------------------------


def _alpaca_bar(day: str, close: float) -> dict:
    return {"t": f"{day}T04:00:00Z", "o": close, "h": close + 1, "l": close - 1,
            "c": close, "v": 1000, "n": 10, "vw": close}  # fmt: skip


@pytest.fixture
def alpaca(monkeypatch: pytest.MonkeyPatch):
    """Point the client at a handler the test sets; record the requests."""
    state = {"handler": None, "requests": []}

    def server(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        return state["handler"](request)

    client = httpx.Client(
        base_url=prices.BASE_URL, transport=httpx.MockTransport(server)
    )
    monkeypatch.setattr(prices, "_client", lambda: client)
    prices._cache.clear()
    yield state
    prices._cache.clear()


def test_bars_follow_page_tokens(alpaca) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("page_token") == "p2":
            body = {"bars": [_alpaca_bar("2026-10-14", 11)], "next_page_token": None}
        else:
            body = {"bars": [_alpaca_bar("2026-10-13", 10)], "next_page_token": "p2"}
        return httpx.Response(200, json={**body, "symbol": "AAPL"})

    alpaca["handler"] = handler
    bars = prices._bars("AAPL", "all", D(2026, 10, 1), D(2026, 10, 14))
    assert [(b.date, b.close, b.high, b.low) for b in bars] == [
        (D(2026, 10, 13), 10, 11, 9),
        (D(2026, 10, 14), 11, 12, 10),
    ]
    params = alpaca["requests"][0].url.params
    assert params["adjustment"] == "all"
    assert params["feed"] == "sip"
    assert params["timeframe"] == "1Day"
    assert params["end"] == "2026-10-14"


def test_unknown_symbol_returns_null_bars(alpaca) -> None:
    alpaca["handler"] = lambda r: httpx.Response(
        200, json={"bars": None, "next_page_token": None, "symbol": "ZZZZQ"}
    )
    with pytest.raises(TickerNotFoundError):
        prices._bars("ZZZZQ", "all", D(2026, 10, 1), D(2026, 10, 14))


@pytest.mark.parametrize(
    ("status", "error", "match"),
    [
        (401, PriceHistoryError, "ALPACA_API_KEY"),
        (403, PriceHistoryError, "ALPACA_API_KEY"),
        (404, TickerNotFoundError, "BRK/B"),
        (429, PriceHistoryError, "rate limit"),
        (500, PriceHistoryError, "HTTP 500"),
    ],
)
def test_http_errors(alpaca, status, error, match) -> None:
    alpaca["handler"] = lambda r: httpx.Response(status, json={"message": "x"})
    with pytest.raises(error, match=match):
        prices._bars("BRK/B", "all", D(2026, 10, 1), D(2026, 10, 14))


def test_connection_error_is_price_history_error(alpaca) -> None:
    def fail(request):
        raise httpx.ConnectError("down")

    alpaca["handler"] = fail
    with pytest.raises(PriceHistoryError, match="down"):
        prices._bars("AAPL", "all", D(2026, 10, 1), D(2026, 10, 14))


# --- fetching and the daily cache --------------------------------------------


@pytest.fixture
def fetches(monkeypatch: pytest.MonkeyPatch):
    """Replace Alpaca and EDGAR with the fixture series; count the calls."""
    calls: list[tuple] = []
    today = [D(2026, 6, 16)]

    def bars(symbol, adjustment, start, end):
        calls.append((symbol, adjustment, start, end))
        return ADJUSTED if adjustment == "all" else SPLIT_ONLY

    monkeypatch.setattr(prices, "_bars", bars)
    monkeypatch.setattr(prices, "_fiscal_year_ends", lambda s: (YEAR_ENDS, None))
    monkeypatch.setattr(prices, "_today", lambda: today[0])
    prices._cache.clear()
    yield calls, today
    prices._cache.clear()


def test_fetches_completed_sessions_only(fetches) -> None:
    calls, _ = fetches
    get_price_history("aapl")
    yesterday = D(2026, 6, 15)
    assert calls == [
        # Five years plus ten days, so the 5y base close exists.
        ("AAPL", "all", D(2021, 6, 5), yesterday),
        ("AAPL", "split", D(2025, 6, 15), yesterday),
    ]


def test_every_period_from_one_fetch_until_the_date_changes(fetches) -> None:
    calls, today = fetches
    one = get_price_history("AAPL", "1y")
    five = get_price_history("AAPL", "5y")
    assert len(calls) == 2
    assert (one.total_return, five.total_return) == pytest.approx((0.25, 1.5))

    today[0] = D(2026, 6, 17)
    get_price_history("AAPL")
    assert len(calls) == 4


def test_no_10k_is_a_note_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_company(symbol, n):
        raise LookupError("not found")

    monkeypatch.setattr(prices, "get_latest_10ks", no_company)
    assert prices._fiscal_year_ends("SPY") == ([], NO_10K)


# --- live ----------------------------------------------------------------------


@pytest.mark.alpaca
def test_live_nvda_split_is_adjusted() -> None:
    # NVDA split 10-for-1 on 2024-06-10. Raw closes drop ~90% overnight;
    # adjusted ones must not, or the split would read as a crash.
    raw = prices._bars("NVDA", "raw", D(2024, 6, 5), D(2024, 6, 12))
    assert min(b.close / a.close for a, b in pairwise(raw)) < 0.2
    for adjustment in ("split", "all"):
        bars = prices._bars("NVDA", adjustment, D(2024, 6, 5), D(2024, 6, 12))
        assert all(0.8 < b.close / a.close < 1.2 for a, b in pairwise(bars))


@pytest.mark.alpaca
@pytest.mark.network
def test_live_rivn_history_is_shorter_than_five_years() -> None:
    h = get_price_history("RIVN", "5y")
    assert not h.full_period
    assert h.start_date == D(2021, 11, 10)
    assert h.week_52_low <= h.week_52_high
    assert all(math.isfinite(fy.total_return) for fy in h.fiscal_year_returns)
    assert 2022 in {fy.fiscal_year for fy in h.fiscal_year_returns}
