"""Live quotes from Finnhub.

Verified against the live API on 2026-10-11 (docs/CONTEXT.md):

* No endpoint carries every field, so a quote takes three calls: ``/quote``
  (price, % change, time), ``/stock/profile2`` (name, market cap) and
  ``/stock/metric`` (52-week range, trailing P/E).
* Symbols are case-sensitive: ``aapl`` returns an all-zero quote.
* An unknown or delisted symbol returns 200 with an all-zero quote (ZZZZ,
  TWTR) or 403 (ZZZQX). Non-US listings are 403 on the free plan.
* ``marketCapitalization`` is in millions, in the company's reporting
  currency: USD for AAPL, TWD for TSM, EUR for ASML, while their US quotes
  are in USD. Read as USD, TSM's would be 30 times too large.
* ``52WeekHigh``/``52WeekLow`` describe the company's primary listing or
  main share class, not the symbol asked for: TSM's are in TWD, TM's in JPY,
  SHEL's in pence (its reporting currency is USD), BRK.B's are class A
  prices. They are kept only when they plausibly belong to the quote; the
  symbol's own range comes from its price history (1.3.24).
* ETFs have no profile (no name or market cap); loss-makers have no P/E.
  Missing values are None, never 0. P/E is a ratio, so it holds whichever
  listing it was computed on.
"""

import datetime as dt
import threading
import time
from functools import lru_cache

import httpx
from pydantic import BaseModel, ConfigDict

from finai.config import settings

BASE_URL = "https://finnhub.io/api/v1"
TIMEOUT_S = 10.0
# The free plan allows 60 calls/min and a quote costs three, so repeated
# questions about one ticker within a minute are answered from the cache.
CACHE_TTL_S = 60.0
# Finnhub reports market capitalisation in millions.
_MILLION = 1_000_000
# A 52-week range is kept only if the price lies within it, give or take
# this margin (the range lags a day, so a breakout can sit just above it).
_RANGE_MARGIN = 0.05


class MarketDataError(RuntimeError):
    """Finnhub could not be reached or answered unexpectedly."""


class TickerNotFoundError(LookupError):
    """Finnhub has no quote for the ticker on the free (US) plan."""


class MarketData(BaseModel):
    """A live quote and the valuation figures around it."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str | None
    price: float
    price_currency: str
    change_pct: float | None
    market_cap: float | None
    market_cap_currency: str | None
    week_52_high: float | None
    week_52_low: float | None
    pe_ttm: float | None
    quote_time: dt.datetime
    source: str = "Finnhub"


@lru_cache(maxsize=1)
def _client() -> httpx.Client:
    """One client per process, so connections are reused across quotes."""
    return httpx.Client(
        base_url=BASE_URL,
        headers={"X-Finnhub-Token": settings.FINNHUB_API_KEY.get_secret_value()},
        timeout=TIMEOUT_S,
    )


def _get(path: str, symbol: str, **params: str) -> dict:
    try:
        response = _client().get(path, params={"symbol": symbol, **params})
    except httpx.HTTPError as error:
        raise MarketDataError(f"Finnhub {path} for {symbol}: {error}") from error
    if response.status_code == 403:
        raise TickerNotFoundError(
            f"no quote for {symbol!r}: unknown, delisted or not a US listing"
        )
    if response.status_code == 429:
        raise MarketDataError("Finnhub rate limit reached (60 calls/min)")
    if response.status_code != 200:
        raise MarketDataError(
            f"Finnhub {path} for {symbol}: HTTP {response.status_code}"
        )
    return response.json()


_cache: dict[str, tuple[float, MarketData]] = {}
_cache_lock = threading.Lock()


def _fetch(symbol: str) -> MarketData:
    quote = _get("/quote", symbol)
    if not quote.get("c") or not quote.get("t"):
        # Unknown and delisted symbols get an all-zero quote, not an error.
        raise TickerNotFoundError(f"no quote for {symbol!r}: unknown or delisted")

    profile = _get("/stock/profile2", symbol)
    metric = _get("/stock/metric", symbol, metric="all").get("metric", {})

    cap_millions = profile.get("marketCapitalization")
    low, high = _own_52_week_range(
        quote["c"],
        profile.get("currency"),
        metric.get("52WeekLow"),
        metric.get("52WeekHigh"),
    )
    return MarketData(
        ticker=symbol,
        name=profile.get("name"),
        price=quote["c"],
        # The free plan only covers US listings, which quote in USD.
        price_currency="USD",
        change_pct=quote.get("dp"),
        market_cap=cap_millions * _MILLION if cap_millions else None,
        market_cap_currency=profile.get("currency") if cap_millions else None,
        week_52_high=high,
        week_52_low=low,
        pe_ttm=metric.get("peTTM"),
        quote_time=dt.datetime.fromtimestamp(quote["t"], dt.UTC),
    )


def _own_52_week_range(
    price: float, currency: str | None, low: float | None, high: float | None
) -> tuple[float | None, float | None]:
    """Return Finnhub's 52-week range only if it belongs to this quote.

    It must be in USD (or come from an ETF, which has no profile and one
    listing) and contain the price. Either test alone lets a wrong range
    through: SHEL reports in USD but its range is in pence, and ASML's EUR
    range happens to sit near its USD price.
    """
    if low is None or high is None:
        return None, None
    if currency not in ("USD", None):
        return None, None
    if not low * (1 - _RANGE_MARGIN) <= price <= high * (1 + _RANGE_MARGIN):
        return None, None
    return low, high


def get_live_market_data(ticker: str) -> MarketData:
    """Return the latest quote for a US-listed ticker, cached for a minute.

    Raises:
        TickerNotFoundError: Finnhub has no quote for it.
        MarketDataError: Finnhub was unreachable, rate-limited or failed.
    """
    symbol = ticker.strip().upper()
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(symbol)
        if cached and now - cached[0] < CACHE_TTL_S:
            return cached[1]
    data = _fetch(symbol)
    with _cache_lock:
        _cache[symbol] = (time.monotonic(), data)
    return data
