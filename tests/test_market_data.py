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
def _metric(
    eps_ttm=None,
    eps_annual=None,
    pe_ttm=None,
    pe_annual=None,
    quarter=None,
    fiscal_year_end=None,
    high=None,
    low=None,
) -> dict:
    series = {}
    if quarter:
        series["quarterly"] = {"eps": [{"period": quarter, "v": 0.0}]}
    if fiscal_year_end:
        series["annual"] = {"eps": [{"period": fiscal_year_end, "v": 0.0}]}
    return {
        "metric": {
            "epsTTM": eps_ttm,
            "epsAnnual": eps_annual,
            "peTTM": pe_ttm,
            "peAnnual": pe_annual,
            "52WeekHigh": high,
            "52WeekLow": low,
        },
        "series": series,
    }


def _profile(name, currency, cap_millions) -> dict:
    return {"name": name, "currency": currency, "marketCapitalization": cap_millions}


def _quote(price, change_pct=0.0) -> dict:
    return {"c": price, "dp": change_pct, "t": 1791576000}


_RESPONSES = {
    "AAPL": {
        "/quote": {"c": 336.64, "d": -3.78, "dp": -1.1104, "t": 1791576000},
        "/stock/profile2": _profile("Apple Inc", "USD", 4912984.843791244),
        "/stock/metric": _metric(
            8.7233, 7.465, 38.1058, 43.862, "2026-06-27", "2025-09-27", 345.34, 243.42
        ),
    },
    # Finnhub's peTTM (28.6483) is at an older price than the quote.
    "NVDA": {
        "/quote": _quote(229.28, -0.5207),
        "/stock/profile2": _profile("NVIDIA Corp", "USD", 5525647.897570095),
        "/stock/metric": _metric(
            7.9108, 4.8979, 28.6483, 46.0214, "2026-07-26", "2026-01-25", 243.37, 164.27
        ),
    },
    # A 17% price move since Finnhub's P/E: still the same share.
    "GME": {
        "/quote": _quote(26.56),
        "/stock/profile2": _profile("GameStop Corp", "USD", 11870.0),
        "/stock/metric": _metric(
            1.508, 0.762, 15.0001, 32.0257, "2026-08-01", "2026-01-31"
        ),
    },
    # TWD EPS per Taiwan share; range from the Taiwan listing.
    "TSM": {
        "/quote": _quote(453.31, -1.0219),
        "/stock/profile2": _profile(
            "Taiwan Semiconductor Manufacturing Co Ltd", "TWD", 66127541.308594
        ),
        "/stock/metric": _metric(
            87.3818, 66.2494, 29.1831, 38.4936, "2026-06-30", "2025-12-31", 2590, 1375
        ),
    },
    # USD reporting, but EPS per ordinary share (2 per US share), range in pence.
    "SHEL": {
        "/quote": _quote(100.18, 0.3),
        "/stock/profile2": _profile("Shell PLC", "USD", 210000.0),
        "/stock/metric": _metric(
            4.5318, 2.9985, 10.981, 15.9879, "2026-06-30", "2025-12-31", 3806.5, 2554
        ),
    },
    # USD, but EPS per class A share.
    "BRK.B": {
        "/quote": _quote(515.62),
        "/stock/profile2": _profile("Berkshire Hathaway Inc", "USD", 993421.589458947),
        "/stock/metric": _metric(
            59668.8094,
            46570.2364,
            11.5827,
            14.8343,
            "2026-06-30",
            "2025-12-31",
            806102.8,
            698000,
        ),
    },
    # EUR EPS: price / EPS lands at 1.10x Finnhub's, inside the agreement band.
    "ASML": {
        "/quote": _quote(1780.34, 1.0),
        "/stock/profile2": _profile("ASML Holding NV", "EUR", 622864.558503),
        "/stock/metric": _metric(
            26.9644, 26.2612, 59.7902, 60.9874, "2026-06-28", "2025-12-31", 1741, 822.2
        ),
    },
    "SPY": {
        "/quote": _quote(778.57, 0.5995),
        "/stock/profile2": {},
        "/stock/metric": _metric(high=781.62, low=629.28),
    },
    "RIVN": {
        "/quote": _quote(13.96, -2.582),
        "/stock/profile2": _profile("Rivian Automotive Inc", "USD", 20212.13414940405),
        "/stock/metric": _metric(
            -2.5821, -3.0742, None, None, "2026-06-30", "2025-12-31", 22.69, 12.39
        ),
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
    assert data.pe_ttm == pytest.approx(336.64 / 8.7233)
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
    assert data.pe_ttm is not None  # company level, from the home listing


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


def test_pe_is_recomputed_from_the_quote(finnhub: _Finnhub) -> None:
    """Finnhub's 28.65 is at an older price; 229.28 / 7.9108 = 28.98."""
    data = get_live_market_data("NVDA")

    assert data.pe_source == market_data.PE_FROM_QUOTE
    assert data.pe_ttm == pytest.approx(229.28 / 7.9108)
    assert data.pe_annual == pytest.approx(229.28 / 4.8979)
    assert data.eps_ttm == 7.9108
    assert data.eps_ttm_period_end == dt.date(2026, 7, 26)
    assert data.eps_annual_period_end == dt.date(2026, 1, 25)


def test_price_move_still_counts_as_the_same_share(finnhub: _Finnhub) -> None:
    data = get_live_market_data("GME")

    assert data.pe_source == market_data.PE_FROM_QUOTE
    assert data.pe_ttm == pytest.approx(26.56 / 1.508)


@pytest.mark.parametrize(
    ("ticker", "why"),
    [
        ("TSM", "EPS in TWD per Taiwan share"),
        ("SHEL", "USD, but EPS per ordinary share, two per US share"),
        ("BRK.B", "USD, but EPS per class A share"),
        ("ASML", "EUR EPS; price / EPS is 1.10x, inside the agreement band"),
    ],
)
def test_other_share_or_currency_falls_back_to_company_level(
    finnhub: _Finnhub, ticker: str, why: str
) -> None:
    data = get_live_market_data(ticker)
    finnhub_pe = _RESPONSES[ticker]["/stock/metric"]["metric"]["peTTM"]

    assert data.pe_source == market_data.PE_COMPANY_LEVEL, why
    assert data.pe_ttm == finnhub_pe
    assert data.eps_ttm is None, why  # not per this ticker's share
    assert data.eps_annual is None


def test_loss_maker_shows_eps_but_no_pe(finnhub: _Finnhub) -> None:
    """A P/E must not become 0; the negative EPS says why there is none."""
    data = get_live_market_data("RIVN")

    assert data.pe_ttm is None
    assert data.pe_annual is None
    assert data.pe_source is None
    assert data.eps_ttm == -2.5821


def test_etf_has_no_valuation(finnhub: _Finnhub) -> None:
    data = get_live_market_data("SPY")

    assert (data.pe_ttm, data.eps_ttm, data.pe_source) == (None, None, None)


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

    nvda = get_live_market_data("NVDA")
    assert nvda.pe_source == market_data.PE_FROM_QUOTE
    assert nvda.pe_ttm == pytest.approx(nvda.price / nvda.eps_ttm)
    assert nvda.eps_ttm_period_end is not None

    aapl = get_live_market_data("AAPL")
    assert aapl.price > 0
    assert aapl.market_cap > 1e12  # trillions, so the millions were converted
    assert aapl.market_cap_currency == "USD"
    assert aapl.quote_time.tzinfo is not None

    tsm = get_live_market_data("TSM")
    assert tsm.market_cap_currency == "TWD"
    assert tsm.week_52_high is None  # Finnhub's is the Taiwan listing's, in TWD
    assert tsm.pe_source == market_data.PE_COMPANY_LEVEL
    with pytest.raises(TickerNotFoundError):
        get_live_market_data("ZZZZ")
