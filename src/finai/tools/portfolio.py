"""The user's portfolio, and what it is worth now.

A stub until task 2.2 reads it from SQLite. It already has the final shape:
holdings as in ``holdings(user_id, ticker, shares, avg_cost)`` (2.2.1), the
same seed data (2.2.2), no database IDs in the output (2.2.6), and Decimal
amounts, since portfolio figures must match Python to the cent (2.2.7).

``user_id`` comes from authentication or the graph config, never from the
request body or an LLM tool argument; the tool wrapper (1.3.4) injects it.

``value_portfolio`` (2.2.5, pulled forward for the verifier, D-049) prices
each holding and computes value and profit or loss in Python, so the model
quotes figures instead of calculating them. Amounts are Decimals rounded to
the cent (half up, as a statement would show them); returns are fractions
(0.064 is 6.4%), as from the other tools (D-036). The verifier recomputes
the same figures independently (``verification.checks.check_portfolio_math``).
"""

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

DEMO_USER_ID = "demo"


class Holding(BaseModel):
    """One position: shares held and the average price paid per share."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    shares: Decimal
    avg_cost: Decimal


class Portfolio(BaseModel):
    """A user's holdings, in USD. Empty when the user holds nothing."""

    model_config = ConfigDict(frozen=True)

    holdings: tuple[Holding, ...]
    currency: str = "USD"


_STUB_HOLDINGS = {
    DEMO_USER_ID: (
        Holding(ticker="AAPL", shares=Decimal(10), avg_cost=Decimal("175.20")),
        Holding(ticker="NVDA", shares=Decimal(25), avg_cost=Decimal("98.50")),
    ),
}


def get_user_portfolio(*, user_id: str) -> Portfolio:
    """Return the holdings of ``user_id``; an unknown user holds nothing.

    Keyword-only, so ``user_id`` is always passed explicitly by code that got
    it from authentication.
    """
    return Portfolio(holdings=_STUB_HOLDINGS.get(user_id, ()))


CENT = Decimal("0.01")
# Returns are fractions; six places keep a percentage exact to 0.0001%.
RATE = Decimal("0.000001")


def cents(amount: Decimal) -> Decimal:
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


class Quote(NamedTuple):
    """A price to value a holding at, and when it was quoted."""

    price: Decimal
    currency: str
    quote_time: dt.datetime


class PositionValue(BaseModel):
    """One holding at the current price. Value fields are None without a price."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    shares: Decimal
    avg_cost: Decimal
    cost_basis: Decimal
    price: Decimal | None
    quote_time: dt.datetime | None
    market_value: Decimal | None
    gain: Decimal | None
    gain_pct: Decimal | None
    note: str | None = None


class PortfolioValue(BaseModel):
    """Every position valued, and the totals over the ones with a price."""

    model_config = ConfigDict(frozen=True)

    positions: tuple[PositionValue, ...]
    total_cost_basis: Decimal
    total_market_value: Decimal
    total_gain: Decimal
    total_gain_pct: Decimal | None
    # False when a position had no usable price; the totals then cover only
    # the positions listed with a market value.
    complete: bool
    currency: str = "USD"


def _value(holding: Holding, quote: Quote | None) -> PositionValue:
    cost = cents(holding.shares * holding.avg_cost)
    common = {
        "ticker": holding.ticker,
        "shares": holding.shares,
        "avg_cost": holding.avg_cost,
        "cost_basis": cost,
    }
    if quote is None or quote.currency != "USD":
        note = "no quote" if quote is None else f"quoted in {quote.currency}, not USD"
        return PositionValue(
            **common,
            price=None,
            quote_time=None,
            market_value=None,
            gain=None,
            gain_pct=None,
            note=note,
        )
    value = cents(holding.shares * quote.price)
    gain = value - cost
    return PositionValue(
        **common,
        price=quote.price,
        quote_time=quote.quote_time,
        market_value=value,
        gain=gain,
        gain_pct=(gain / cost).quantize(RATE, rounding=ROUND_HALF_UP) if cost else None,
    )


def value_portfolio(portfolio: Portfolio, quotes: dict[str, Quote]) -> PortfolioValue:
    """Value each holding at its quote; total the ones that have a price.

    Each position is rounded to the cent first and the totals are sums of
    those, so the parts add up to the total exactly as displayed.
    """
    positions = tuple(_value(h, quotes.get(h.ticker)) for h in portfolio.holdings)
    priced = [p for p in positions if p.market_value is not None]
    cost = sum((p.cost_basis for p in priced), Decimal(0))
    value = sum((p.market_value for p in priced), Decimal(0))
    gain = value - cost
    return PortfolioValue(
        positions=positions,
        total_cost_basis=cost,
        total_market_value=value,
        total_gain=gain,
        total_gain_pct=(gain / cost).quantize(RATE, rounding=ROUND_HALF_UP)
        if cost
        else None,
        complete=len(priced) == len(positions),
        currency=portfolio.currency,
    )
