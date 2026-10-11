"""The user's portfolio.

A stub until task 2.2 reads it from SQLite. It already has the final shape:
holdings as in ``holdings(user_id, ticker, shares, avg_cost)`` (2.2.1), the
same seed data (2.2.2), no database IDs in the output (2.2.6), and Decimal
amounts, since portfolio figures must match Python to the cent (2.2.7).

``user_id`` comes from authentication or the graph config, never from the
request body or an LLM tool argument; the tool wrapper (1.3.4) injects it.
"""

from decimal import Decimal

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
