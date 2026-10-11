"""Tests for live market data.

Offline tests replay response shapes recorded from Finnhub on 2026-10-11
through httpx's mock transport. ``-m finnhub`` runs the ones that call the API.
"""

import datetime as dt

import httpx
import pytest

from finai.tools import market_data
from finai.tools.market_data import (
    MarketDataError,
    TickerNotFoundError,
    get_live_market_data,
)

_ZERO_QUOTE = {"c": 0, "d": None, "dp": None, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0}

# Recorded responses, trimmed to the fields the tool reads.
_RESPONSES = {
    "AAPL": {
        "/quote": {"c": 336.64, "d": -3.78, "dp": -1.1104, "t": 1791576000},
        "/stock/profile2": {
            "name": "Apple Inc",
            "currency": "USD",
            "marketCapitalization": 4912984.843791244,
        },
        "/stock/metric": {
            "metric": {"52WeekHigh": 345.34, "52WeekLow": 243.42, "peTTM": 38.1058}
        },
    },
    "TSM": {
        "/quote": {"c": 453.31, "dp": -1.0219, "t": 1791576000},
        "/stock/profile2": {
            "name": "Taiwan Semiconductor Manufacturing Co Ltd",
            "currency": "TWD",
            "marketCapitalization": 66127541.308594,
        },
        "/stock/metric": {
            "metric": {"52WeekHigh": 2590, "52WeekLow": 1375, "peTTM": 29.1831}
        },
    },
    "SPY": {
        "/quote": {"c": 778.57, "dp": 0.5995, "t": 1791576000},
        "/stock/profile2": {},
        "/stock/metric": {"metric": {"52WeekHigh": 781.62, "52WeekLow": 629.28}},
    },
    "RIVN": {
        "/quote": {"c": 13.96, "dp": -2.582, "t": 1791576000},
        "/stock/profile2": {
            "name": "Rivian Automotive Inc",
            "currency": "USD",
            "marketCapitalization": 20212.13414940405,
        },
        "/stock/metric": {"metric": {"52WeekHigh": 22.69, "peTTM": None}},
    },
    # SHEL reports in USD, but its range is London's, in pence.
    "SHEL": {
        "/quote": {"c": 100.18, "dp": 0.3, "t": 1791576000},
        "/stock/profile2": {
            "name": "Shell PLC",
            "currency": "USD",
            "marketCapitalization": 210000.0,
        },
        "/stock/metric": {
            "metric": {"52WeekHigh": 3806.5, "52WeekLow": 2554, "peTTM": 12.1}
        },
    },
    # ASML's EUR range sits near its USD price by coincidence.
    "ASML": {
        "/quote": {"c": 1780.34, "dp": 1.0, "t": 1791576000},
        "/stock/profile2": {
            "name": "ASML Holding NV",
            "currency": "EUR",
            "marketCapitalization": 622864.558503,
        },
        "/stock/metric": {
            "metric": {"52WeekHigh": 1741, "52WeekLow": 822.2, "peTTM": 59.79}
        },
    },
    "ZZZZ": {"/quote": _ZERO_QUOTE, "/stock/profile2": {}, "/stock/metric": {}},
}


class _Finnhub:
    """Serves recorded responses and records each request."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        symbol = request.url.params["symbol"]
        self.requests.append((request.url.path.removeprefix("/api/v1"), symbol))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "nope"})
        if symbol == "ZZZQX":
            return httpx.Response(
                403, json={"error": "You don't have access to this resource."}
            )
        path = request.url.path.removeprefix("/api/v1")
        # Finnhub is case-sensitive: an unknown spelling gets a zero quote.
        body = _RESPONSES.get(symbol, _RESPONSES["ZZZZ"])[path]
        return httpx.Response(200, json=body)


@pytest.fixture
def finnhub(monkeypatch: pytest.MonkeyPatch) -> _Finnhub:
    server = _Finnhub()
    client = httpx.Client(
        base_url=market_data.BASE_URL, transport=httpx.MockTransport(server)
    )
    monkeypatch.setattr(market_data, "_client", lambda: client)
    market_data._cache.clear()
    yield server
    market_data._cache.clear()


def test_quote_maps_every_field(finnhub: _Finnhub) -> None:
    data = get_live_market_data("AAPL")

    assert data.name == "Apple Inc"
    assert data.price == 336.64
    assert data.price_currency == "USD"
    assert data.change_pct == -1.1104
    assert data.week_52_high == 345.34
    assert data.week_52_low == 243.42
    assert data.pe_ttm == 38.1058
    assert data.quote_time == dt.datetime(2026, 10, 9, 20, 0, tzinfo=dt.UTC)
    assert data.source == "Finnhub"


def test_market_cap_is_converted_from_millions(finnhub: _Finnhub) -> None:
    """Finnhub's 4,912,984.84 means $4.91 trillion, not $4.9 million."""
    data = get_live_market_data("AAPL")

    assert data.market_cap == pytest.approx(4.912984843791244e12)
    assert data.market_cap_currency == "USD"


def test_market_cap_keeps_its_reporting_currency(finnhub: _Finnhub) -> None:
    """TSM's cap is in TWD; read as USD it would be about 30 times too large."""
    data = get_live_market_data("TSM")

    assert data.price_currency == "USD"
    assert data.market_cap_currency == "TWD"


@pytest.mark.parametrize(
    ("ticker", "why"),
    [
        ("TSM", "range in TWD from the Taiwan listing"),
        ("SHEL", "USD reporter, but range in pence from London"),
        ("ASML", "EUR range that happens to sit near the USD price"),
    ],
)
def test_52_week_range_of_another_listing_is_dropped(
    finnhub: _Finnhub, ticker: str, why: str
) -> None:
    data = get_live_market_data(ticker)

    assert data.week_52_low is None, why
    assert data.week_52_high is None, why
    assert data.pe_ttm is not None  # a ratio, valid on any listing


def test_etf_keeps_its_52_week_range(finnhub: _Finnhub) -> None:
    """ETFs have no profile, hence no currency, and one listing."""
    data = get_live_market_data("SPY")

    assert (data.week_52_low, data.week_52_high) == (629.28, 781.62)


def test_lowercase_ticker_is_upper_cased(finnhub: _Finnhub) -> None:
    """Finnhub returns an all-zero quote for "aapl"."""
    assert get_live_market_data(" aapl ").ticker == "AAPL"
    assert {symbol for _, symbol in finnhub.requests} == {"AAPL"}


def test_etf_has_no_name_or_market_cap(finnhub: _Finnhub) -> None:
    data = get_live_market_data("SPY")

    assert data.price == 778.57
    assert data.name is None
    assert data.market_cap is None
    assert data.market_cap_currency is None
    assert data.week_52_high == 781.62


def test_missing_pe_stays_none(finnhub: _Finnhub) -> None:
    """A loss-maker has no P/E; it must not become 0."""
    assert get_live_market_data("RIVN").pe_ttm is None


def test_zero_quote_means_not_found_and_stops_early(finnhub: _Finnhub) -> None:
    with pytest.raises(TickerNotFoundError, match="ZZZZ"):
        get_live_market_data("ZZZZ")

    assert finnhub.requests == [("/quote", "ZZZZ")]


def test_403_means_not_found(finnhub: _Finnhub) -> None:
    with pytest.raises(TickerNotFoundError, match="ZZZQX"):
        get_live_market_data("ZZZQX")


@pytest.mark.parametrize("status", [429, 500])
def test_server_errors_raise_market_data_error(finnhub: _Finnhub, status: int) -> None:
    finnhub.status = status

    with pytest.raises(MarketDataError):
        get_live_market_data("AAPL")


def test_network_failure_raises_market_data_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(
        base_url=market_data.BASE_URL, transport=httpx.MockTransport(fail)
    )
    monkeypatch.setattr(market_data, "_client", lambda: client)
    market_data._cache.clear()

    with pytest.raises(MarketDataError, match="connection refused"):
        get_live_market_data("AAPL")


def test_three_calls_per_quote_then_cached(finnhub: _Finnhub) -> None:
    first = get_live_market_data("AAPL")
    second = get_live_market_data("aapl")

    assert second is first
    assert len(finnhub.requests) == 3


def test_cache_expires(finnhub: _Finnhub, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(market_data.time, "monotonic", lambda: clock[0])

    get_live_market_data("AAPL")
    clock[0] += market_data.CACHE_TTL_S + 1
    get_live_market_data("AAPL")

    assert len(finnhub.requests) == 6


@pytest.mark.finnhub
def test_live_quotes() -> None:
    market_data._client.cache_clear()
    market_data._cache.clear()

    aapl = get_live_market_data("AAPL")
    assert aapl.price > 0
    assert aapl.market_cap > 1e12  # trillions, so the millions were converted
    assert aapl.market_cap_currency == "USD"
    assert aapl.quote_time.tzinfo is not None

    tsm = get_live_market_data("TSM")
    assert tsm.market_cap_currency == "TWD"
    assert tsm.week_52_high is None  # Finnhub's is the Taiwan listing's, in TWD
    with pytest.raises(TickerNotFoundError):
        get_live_market_data("ZZZZ")
