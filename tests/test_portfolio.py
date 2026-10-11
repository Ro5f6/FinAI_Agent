"""Tests for the portfolio stub."""

from decimal import Decimal

import pytest

from finai.tools.portfolio import DEMO_USER_ID, Holding, get_user_portfolio


def test_demo_user_holds_the_seed_positions() -> None:
    """The same positions 2.2.2 seeds into SQLite."""
    portfolio = get_user_portfolio(user_id=DEMO_USER_ID)

    assert portfolio.holdings == (
        Holding(ticker="AAPL", shares=Decimal(10), avg_cost=Decimal("175.20")),
        Holding(ticker="NVDA", shares=Decimal(25), avg_cost=Decimal("98.50")),
    )
    assert portfolio.currency == "USD"


def test_amounts_are_exact_decimals() -> None:
    """Cost basis must come out to the cent: 10 x 175.20 + 25 x 98.50."""
    holdings = get_user_portfolio(user_id=DEMO_USER_ID).holdings

    assert all(isinstance(h.avg_cost, Decimal) for h in holdings)
    assert sum(h.shares * h.avg_cost for h in holdings) == Decimal("4214.50")


def test_unknown_user_gets_an_empty_portfolio() -> None:
    """Never another user's holdings."""
    assert get_user_portfolio(user_id="someone-else").holdings == ()


def test_user_id_must_be_passed_by_keyword() -> None:
    with pytest.raises(TypeError):
        get_user_portfolio(DEMO_USER_ID)  # type: ignore[misc]


def test_output_carries_no_database_ids() -> None:
    """2.2.6: tool output must not expose internal IDs."""
    portfolio = get_user_portfolio(user_id=DEMO_USER_ID)

    assert set(portfolio.model_dump()) == {"holdings", "currency"}
    assert set(portfolio.holdings[0].model_dump()) == {"ticker", "shares", "avg_cost"}
