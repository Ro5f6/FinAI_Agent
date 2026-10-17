"""Price history from Alpaca daily bars (1.3.24).

Verified against the live API on 2026-10-15 (docs/CONTEXT.md):

* ``adjustment`` takes ``raw``, ``split`` or ``all``. Raw NVDA closes fall from
  $1,208.88 to $121.79 across the June 2024 10-for-1 split; ``split`` and
  ``all`` are continuous. ``all`` also folds dividends into past prices, so a
  close-to-close return on it is a total return: XOM's year is 63.2% on
  ``all`` against 58.6% on ``split``.
* Dividend adjustment also lowers past highs and lows (T's 52-week high by
  3.5%), so the 52-week range comes from ``split`` bars, the prices the stock
  actually traded at, and returns from ``all`` bars. Two calls per ticker.
* The free plan serves the full-market SIP feed for history. A request that
  reaches into the latest 15 minutes is refused (403), and during the session
  today's bar is partial, so only completed sessions are requested: bars up
  to yesterday, New York time. The cache then lasts until the date changes.
* Symbols are case-insensitive; share classes use a dot (``BRK.B``; ``BRK/B``
  is 404). An unknown symbol returns 200 with ``"bars": null``.
* Bar times are New York midnight in UTC (``04:00Z`` or ``05:00Z``).
* One page holds up to 10,000 bars; five years is ~1,260.

Definitions:

* **Total return** is the last close over the base close, minus one. The base
  is the last close on or before the period's start date, so a year runs
  close to close like a fund's reported return.
* **Annualised volatility** is the sample standard deviation of daily log
  returns times the square root of 252 trading days.
* **Maximum drawdown** is the largest fall from a running peak close to a
  later close, as a negative fraction, with both dates.
* **Fiscal-year returns** run from one 10-K's ``period_of_report`` to the
  next, so price and fundamentals (1.3.22) cover the same year even when it
  ends in September (AAPL) or June (MSFT).
"""

import datetime as dt
import math
import statistics
import threading
from functools import lru_cache
from itertools import pairwise
from typing import Literal, NamedTuple
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict

from finai.config import settings
from finai.ingestion.fetch import get_latest_10ks
from finai.tools.market_data import TickerNotFoundError

BASE_URL = "https://data.alpaca.markets"
TIMEOUT_S = 15.0
NEW_YORK = ZoneInfo("America/New_York")
TRADING_DAYS = 252
PAGE_LIMIT = 10_000

Period = Literal["1y", "3y", "5y"]
PERIOD_YEARS: dict[str, int] = {"1y": 1, "3y": 3, "5y": 5}
FETCH_YEARS = 5
# Extra calendar days fetched before the five-year start, so a base close on
# or before it exists even after a long weekend.
_FETCH_MARGIN_DAYS = 10
# 10-Ks read for fiscal-year boundaries: six year ends bound five years.
_FISCAL_FILINGS = 6

SOURCE = "Alpaca daily bars, SIP (all US exchanges)"
NO_10K = "no 10-K on EDGAR (an ETF, or a foreign filer on Form 20-F)"


class PriceHistoryError(RuntimeError):
    """Alpaca could not be reached or answered unexpectedly."""


class Bar(NamedTuple):
    date: dt.date
    high: float
    low: float
    close: float


class FiscalYearReturn(BaseModel):
    """The stock's total return over one fiscal year of the company's 10-K."""

    model_config = ConfigDict(frozen=True)

    fiscal_year: int
    # The fiscal year's end dates; the closes used are the last on or before.
    start_date: dt.date
    end_date: dt.date
    total_return: float


class PriceHistory(BaseModel):
    """Return, risk and range for one period, from adjusted daily closes."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    period: str
    # The base close's date. Later than requested_start when the listing is
    # younger than the period (RIVN's five years start in November 2021).
    requested_start: dt.date
    start_date: dt.date
    end_date: dt.date
    full_period: bool
    # Fractions: 0.064 = 6.4%. Dividend-adjusted, so total returns.
    start_close: float
    end_close: float
    total_return: float
    # Per year, for the 3y and 5y periods; None for 1y, where it would only
    # repeat total_return.
    annualised_return: float | None
    annualised_volatility: float | None
    max_drawdown: float
    max_drawdown_peak: dt.date
    max_drawdown_trough: dt.date
    # Newest first; only years fully inside the five years of bars.
    fiscal_year_returns: list[FiscalYearReturn]
    fiscal_years_note: str | None
    # From split-adjusted highs and lows over the year to end_date: the
    # prices the stock traded at, unlike the dividend-adjusted closes above.
    week_52_high: float
    week_52_high_date: dt.date
    week_52_low: float
    week_52_low_date: dt.date
    source: str = SOURCE


@lru_cache(maxsize=1)
def _client() -> httpx.Client:
    """One client per process, so connections are reused across tickers."""
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "APCA-API-KEY-ID": settings.ALPACA_API_KEY.get_secret_value(),
            "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY.get_secret_value(),
        },
        timeout=TIMEOUT_S,
    )


def _bar_date(timestamp: str) -> dt.date:
    """The trading day of a bar stamped at New York midnight in UTC."""
    utc = dt.datetime.fromisoformat(timestamp)
    return utc.astimezone(NEW_YORK).date()


def _bars(symbol: str, adjustment: str, start: dt.date, end: dt.date) -> list[Bar]:
    """Daily bars from ``start`` to ``end`` inclusive, oldest first.

    Raises:
        TickerNotFoundError: Alpaca has no bars for the symbol.
        PriceHistoryError: Alpaca was unreachable, refused or failed.
    """
    params = {
        "timeframe": "1Day",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": adjustment,
        "feed": "sip",
        "limit": str(PAGE_LIMIT),
    }
    bars: list[Bar] = []
    page_token = None
    while True:
        page = {**params, "page_token": page_token} if page_token else params
        try:
            response = _client().get(f"/v2/stocks/{symbol}/bars", params=page)
        except httpx.HTTPError as error:
            raise PriceHistoryError(f"Alpaca bars for {symbol}: {error}") from error
        if response.status_code in (401, 403):
            raise PriceHistoryError(
                f"Alpaca refused bars for {symbol} (HTTP {response.status_code}): "
                "check ALPACA_API_KEY and ALPACA_SECRET_KEY"
            )
        if response.status_code == 404:
            raise TickerNotFoundError(f"no price history for {symbol!r}")
        if response.status_code == 429:
            raise PriceHistoryError("Alpaca rate limit reached (200 calls/min)")
        if response.status_code != 200:
            raise PriceHistoryError(
                f"Alpaca bars for {symbol}: HTTP {response.status_code}"
            )
        body = response.json()
        bars.extend(
            Bar(_bar_date(b["t"]), float(b["h"]), float(b["l"]), float(b["c"]))
            for b in body.get("bars") or []
        )
        page_token = body.get("next_page_token")
        if not page_token:
            break
    if not bars:
        # Unknown symbols return 200 with "bars": null, not an error.
        raise TickerNotFoundError(f"no price history for {symbol!r}")
    return bars


def _years_before(date: dt.date, years: int) -> dt.date:
    try:
        return date.replace(year=date.year - years)
    except ValueError:  # 29 February
        return date.replace(year=date.year - years, day=28)


def _last_on_or_before(bars: list[Bar], date: dt.date) -> int | None:
    """Index of the last bar dated ``date`` or earlier."""
    index = None
    for i, bar in enumerate(bars):
        if bar.date > date:
            break
        index = i
    return index


def _volatility(closes: list[float]) -> float | None:
    if len(closes) < 3:
        return None
    log_returns = [math.log(b / a) for a, b in pairwise(closes)]
    return statistics.stdev(log_returns) * math.sqrt(TRADING_DAYS)


def _max_drawdown(bars: list[Bar]) -> tuple[float, dt.date, dt.date]:
    """The largest peak-to-later-trough fall in close, as a fraction <= 0."""
    peak = bars[0]
    worst = (0.0, bars[0].date, bars[0].date)
    for bar in bars:
        if bar.close > peak.close:
            peak = bar
        drawdown = bar.close / peak.close - 1
        if drawdown < worst[0]:
            worst = (drawdown, peak.date, bar.date)
    return worst


def _fiscal_year_returns(
    bars: list[Bar], year_ends: list[dt.date]
) -> list[FiscalYearReturn]:
    """Returns between consecutive fiscal-year ends inside ``bars``, newest first."""
    results = []
    for start, end in pairwise(sorted(year_ends)):
        # A year that starts before the first bar would be measured from a
        # later close; one that ends after the last bar isn't over.
        if start < bars[0].date or end > bars[-1].date:
            continue
        first = _last_on_or_before(bars, start)
        last = _last_on_or_before(bars, end)
        if first is None or last is None:
            continue
        results.append(
            FiscalYearReturn(
                fiscal_year=end.year,
                start_date=start,
                end_date=end,
                total_return=bars[last].close / bars[first].close - 1,
            )
        )
    return sorted(results, key=lambda r: r.end_date, reverse=True)


def compute_price_history(
    ticker: str,
    period: Period,
    adjusted: list[Bar],
    split_only: list[Bar],
    fiscal_year_ends: list[dt.date],
    fiscal_years_note: str | None = None,
) -> PriceHistory:
    """Figures for ``period`` from bars already fetched, oldest first.

    ``adjusted`` are split- and dividend-adjusted bars, ``split_only`` split-
    adjusted bars over at least the last year.
    """
    if period not in PERIOD_YEARS:
        raise ValueError(
            f"period must be one of {sorted(PERIOD_YEARS)}, got {period!r}"
        )
    end = adjusted[-1]
    requested_start = _years_before(end.date, PERIOD_YEARS[period])
    base = _last_on_or_before(adjusted, requested_start)
    full_period = base is not None
    window = adjusted[base or 0 :]
    first = window[0]

    total = end.close / first.close - 1
    annualised = None
    if PERIOD_YEARS[period] > 1:
        years = (end.date - first.date).days / 365.25
        annualised = (1 + total) ** (1 / years) - 1 if total > -1 else None
    drawdown, peak, trough = _max_drawdown(window)

    year_ago = _years_before(end.date, 1)
    last_year = [b for b in split_only if year_ago < b.date <= end.date]
    high = max(last_year, key=lambda b: b.high)
    low = min(last_year, key=lambda b: b.low)

    fiscal = _fiscal_year_returns(adjusted, fiscal_year_ends)
    if not fiscal and fiscal_years_note is None:
        fiscal_years_note = "no complete fiscal year inside the price history"

    return PriceHistory(
        ticker=ticker,
        period=period,
        requested_start=requested_start,
        start_date=first.date,
        end_date=end.date,
        full_period=full_period,
        start_close=first.close,
        end_close=end.close,
        total_return=total,
        annualised_return=annualised,
        annualised_volatility=_volatility([b.close for b in window]),
        max_drawdown=drawdown,
        max_drawdown_peak=peak,
        max_drawdown_trough=trough,
        fiscal_year_returns=fiscal,
        fiscal_years_note=fiscal_years_note,
        week_52_high=high.high,
        week_52_high_date=high.date,
        week_52_low=low.low,
        week_52_low_date=low.date,
    )


class _Fetched(NamedTuple):
    requested_on: dt.date
    adjusted: list[Bar]
    split_only: list[Bar]
    fiscal_year_ends: list[dt.date]
    fiscal_years_note: str | None


_cache: dict[str, _Fetched] = {}
_cache_lock = threading.Lock()


def _today() -> dt.date:
    return dt.datetime.now(NEW_YORK).date()


def _fiscal_year_ends(symbol: str) -> tuple[list[dt.date], str | None]:
    try:
        filings = get_latest_10ks(symbol, _FISCAL_FILINGS)
    except LookupError:
        # UnknownTickerError and FilingNotFoundError from fetch.resolve_ticker.
        return [], NO_10K
    return [dt.date.fromisoformat(str(f.period_of_report)) for f in filings], None


def _fetch(symbol: str, today: dt.date) -> _Fetched:
    # Completed sessions only: today's bar is partial until the close.
    end = today - dt.timedelta(days=1)
    start = _years_before(end, FETCH_YEARS) - dt.timedelta(days=_FETCH_MARGIN_DAYS)
    adjusted = _bars(symbol, "all", start, end)
    split_only = _bars(symbol, "split", _years_before(end, 1), end)
    ends, note = _fiscal_year_ends(symbol)
    return _Fetched(today, adjusted, split_only, ends, note)


def get_price_history(ticker: str, period: Period = "1y") -> PriceHistory:
    """Return, volatility, drawdown and 52-week range for ``ticker``.

    Five years of bars are fetched once a day per ticker and every period is
    sliced from them, so "1y" and "5y" cost the same.

    Raises:
        ValueError: ``period`` is not "1y", "3y" or "5y".
        TickerNotFoundError: Alpaca has no bars for the ticker.
        PriceHistoryError: Alpaca was unreachable, refused or failed.
    """
    if period not in PERIOD_YEARS:
        raise ValueError(
            f"period must be one of {sorted(PERIOD_YEARS)}, got {period!r}"
        )
    symbol = ticker.strip().upper()
    today = _today()
    with _cache_lock:
        fetched = _cache.get(symbol)
    if fetched is None or fetched.requested_on != today:
        fetched = _fetch(symbol, today)
        with _cache_lock:
            _cache[symbol] = fetched
    return compute_price_history(
        symbol,
        period,
        fetched.adjusted,
        fetched.split_only,
        fetched.fiscal_year_ends,
        fetched.fiscal_years_note,
    )
