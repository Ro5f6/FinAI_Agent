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


# --- valuation (2.2.5, pulled forward for 1.5.6) -----------------------------------

import datetime as dt

from finai.tools.portfolio import Portfolio, Quote, value_portfolio

_T = dt.datetime(2026, 10, 22, 20, 0, tzinfo=dt.UTC)


def _quote(price: str, currency: str = "USD") -> Quote:
    return Quote(Decimal(price), currency, _T)


def test_valuation_by_hand() -> None:
    """AAPL 10 x 175.20 = 1,752.00 cost, 10 x 258.45 = 2,584.50 value;
    NVDA 25 x 98.50 = 2,462.50 cost, 25 x 212.21 = 5,305.25 value."""
    value = value_portfolio(
        get_user_portfolio(user_id=DEMO_USER_ID),
        {"AAPL": _quote("258.45"), "NVDA": _quote("212.21")},
    )
    aapl, nvda = value.positions
    assert (aapl.cost_basis, aapl.market_value, aapl.gain) == (
        Decimal("1752.00"),
        Decimal("2584.50"),
        Decimal("832.50"),
    )
    assert aapl.gain_pct == Decimal("0.475171")  # 832.50 / 1752.00
    assert (nvda.market_value, nvda.gain) == (Decimal("5305.25"), Decimal("2842.75"))
    assert (value.total_cost_basis, value.total_market_value, value.total_gain) == (
        Decimal("4214.50"),
        Decimal("7889.75"),
        Decimal("3675.25"),
    )
    assert value.total_gain_pct == Decimal("0.872049")  # 3675.25 / 4214.50
    assert value.complete


def test_positions_are_rounded_before_totalling() -> None:
    """Parts add up to the total as shown: 3 x 0.335 = 1.005 -> 1.01 each."""
    holdings = Portfolio(
        holdings=(
            Holding(ticker="A", shares=Decimal(3), avg_cost=Decimal("0.10")),
            Holding(ticker="B", shares=Decimal(3), avg_cost=Decimal("0.10")),
        )
    )
    value = value_portfolio(holdings, {"A": _quote("0.335"), "B": _quote("0.335")})
    assert [p.market_value for p in value.positions] == [Decimal("1.01")] * 2
    assert value.total_market_value == Decimal("2.02")


def test_a_loss_is_a_negative_gain() -> None:
    value = value_portfolio(
        get_user_portfolio(user_id=DEMO_USER_ID), {"NVDA": _quote("90.00")}
    )
    nvda = value.positions[1]
    assert nvda.gain == Decimal("-212.50")  # 2,250.00 - 2,462.50


def test_a_holding_without_a_usable_price_is_left_out_of_the_totals() -> None:
    value = value_portfolio(
        get_user_portfolio(user_id=DEMO_USER_ID),
        {"AAPL": _quote("258.45", currency="EUR")},  # NVDA: no quote at all
    )
    aapl, nvda = value.positions
    assert (aapl.market_value, aapl.note) == (None, "quoted in EUR, not USD")
    assert (nvda.market_value, nvda.note) == (None, "no quote")
    assert not value.complete
    assert value.total_market_value == Decimal(0)
    assert value.total_gain_pct is None
