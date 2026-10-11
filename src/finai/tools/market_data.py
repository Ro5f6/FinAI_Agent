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
* ``peTTM`` is computed at an older price than the quote (NVDA: 28.65 at
  ~$226.6 while the quote is $229.28), so P/E is recomputed here as price /
  EPS. That is valid only when both are per the same share in the same
  currency: TSM's EPS is TWD per Taiwan share, SHEL's per ordinary share (two
  per US share), BRK.B's per class A share. Otherwise Finnhub's own P/E,
  computed on the home listing or main class, is returned as company-level.
* ETFs have no profile (no name or market cap) and no EPS; loss-makers have
  negative EPS and no P/E. Missing values are None, never 0.
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
# Price / EPS is trusted when it lands within this factor of Finnhub's P/E:
# US common stocks measured 1.00-1.03, GME 1.17 after a price move; a
# different share or ADR ratio gives 2.01 (SHEL), 0.18 (TSM), 0.00 (BRK.B).
_PE_AGREEMENT = 1.33

PE_FROM_QUOTE = "price / EPS"
PE_COMPANY_LEVEL = "Finnhub, company level (home listing or main share class)"


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
    # P/E on trailing-twelve-month and last-fiscal-year EPS. When pe_source is
    # PE_FROM_QUOTE they are price / eps_*, computed here from this quote;
    # otherwise Finnhub's and the EPS fields are None.
    pe_ttm: float | None
    pe_annual: float | None
    pe_source: str | None
    eps_ttm: float | None
    eps_ttm_period_end: dt.date | None
    eps_annual: float | None
    eps_annual_period_end: dt.date | None
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
    metrics = _get("/stock/metric", symbol, metric="all")
    metric = metrics.get("metric", {})

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
        **_valuation(
            quote["c"], profile.get("currency"), metric, metrics.get("series", {})
        ),
        quote_time=dt.datetime.fromtimestamp(quote["t"], dt.UTC),
    )


def _latest_period(series: dict, frequency: str) -> dt.date | None:
    """End date of the most recent EPS period in Finnhub's series."""
    points = series.get(frequency, {}).get("eps") or []
    return dt.date.fromisoformat(points[0]["period"]) if points else None


def _valuation(price: float, currency: str | None, metric: dict, series: dict) -> dict:
    """P/E and EPS fields for MarketData; arithmetic stays in Python.

    Price / EPS is used when the company reports in USD and the result agrees
    with Finnhub's P/E within ``_PE_AGREEMENT``. The currency test catches
    ASML (EUR EPS, ratio 1.10, inside the band); the agreement test catches
    SHEL and BRK.B (USD, but another share). Otherwise Finnhub's P/E is
    returned as company-level with no EPS, since its EPS isn't per this
    ticker's share.
    """
    eps_ttm, eps_annual = metric.get("epsTTM"), metric.get("epsAnnual")
    finnhub_ttm, finnhub_annual = metric.get("peTTM"), metric.get("peAnnual")

    def per_share(eps: float | None) -> float | None:
        return price / eps if eps and eps > 0 else None

    computed = per_share(eps_ttm)
    same_share = (
        currency == "USD"
        and computed is not None
        and finnhub_ttm is not None
        and 1 / _PE_AGREEMENT <= computed / finnhub_ttm <= _PE_AGREEMENT
    )
    loss_maker = currency == "USD" and eps_ttm is not None and eps_ttm <= 0
    if same_share or loss_maker:
        return {
            "pe_ttm": computed,
            "pe_annual": per_share(eps_annual),
            "pe_source": PE_FROM_QUOTE if computed is not None else None,
            "eps_ttm": eps_ttm,
            "eps_ttm_period_end": _latest_period(series, "quarterly"),
            "eps_annual": eps_annual,
            "eps_annual_period_end": _latest_period(series, "annual"),
        }
    company_level = finnhub_ttm is not None or finnhub_annual is not None
    return {
        "pe_ttm": finnhub_ttm,
        "pe_annual": finnhub_annual,
        "pe_source": PE_COMPANY_LEVEL if company_level else None,
        "eps_ttm": None,
        "eps_ttm_period_end": None,
        "eps_annual": None,
        "eps_annual_period_end": None,
    }


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
