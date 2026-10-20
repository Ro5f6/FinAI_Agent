"""Tests for the LangChain tool wrappers: what the model sees, and that
user_id comes only from the run's config."""

import datetime as dt
import typing

import pytest

from finai.agent.prompts import FILING_CITATION, PRICE_CITATION
from finai.ingestion.parse import TARGET_ITEMS
from finai.tools import agent_tools, filings, financials, market_data, prices
from finai.tools.agent_tools import (
    TOOLS,
    MissingUserError,
    get_financial_history,
    get_live_market_data,
    get_price_history,
    get_user_portfolio,
    query_company_10k,
)
from finai.tools.filings import TenKSearchResult
from finai.tools.financials import Figure, FinancialHistory
from finai.tools.trends import NOT_MEANINGFUL


def _schema(tool) -> dict:
    return tool.tool_call_schema.model_json_schema()


def test_the_tools_are_registered() -> None:
    assert [t.name for t in TOOLS] == [
        "get_live_market_data",
        "query_company_10k",
        "get_user_portfolio",
        "get_financial_history",
        "get_price_history",
    ]


def test_every_argument_is_described() -> None:
    """The model's only guide to an argument is its description."""
    for tool in TOOLS:
        assert tool.description
        for name, prop in _schema(tool).get("properties", {}).items():
            assert prop.get("description"), f"{tool.name}.{name}"


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_no_tool_lets_the_model_choose_the_user(tool) -> None:
    properties = _schema(tool).get("properties", {})

    assert "user_id" not in properties
    assert "config" not in properties


def test_10k_search_without_a_user_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    users = []

    def fake(*args, user_id):
        users.append(user_id)
        return TenKSearchResult(
            status="not_indexed", ticker="PLTR", fiscal_year=None, section=None,
            available_years=[],
        )  # fmt: skip

    monkeypatch.setattr(filings, "query_company_10k", fake)
    query_company_10k.invoke({"ticker": "PLTR", "question": "risks"})
    assert users == [None]


def test_required_arguments() -> None:
    assert _schema(get_live_market_data)["required"] == ["ticker"]
    assert _schema(query_company_10k)["required"] == ["ticker", "question"]
    assert _schema(get_user_portfolio).get("required", []) == []
    assert _schema(get_financial_history)["required"] == ["ticker"]
    assert _schema(get_price_history)["required"] == ["ticker"]


def test_price_period_is_an_enum_defaulting_to_one_year() -> None:
    period = _schema(get_price_history)["properties"]["period"]
    assert period["enum"] == ["1y", "3y", "5y"]
    assert period["default"] == "1y"


def test_section_values_match_the_ingested_items() -> None:
    assert typing.get_args(agent_tools.Section) == TARGET_ITEMS


def test_portfolio_user_comes_from_the_config() -> None:
    result = get_user_portfolio.invoke({}, config={"configurable": {"user_id": "demo"}})

    assert [h["ticker"] for h in result["holdings"]] == ["AAPL", "NVDA"]
    assert result["holdings"][0]["avg_cost"] == "175.20"


def test_portfolio_ignores_a_user_the_model_supplies() -> None:
    """A user_id or config sent as a tool argument must not change the user."""
    result = get_user_portfolio.invoke(
        {"user_id": "demo", "config": {"configurable": {"user_id": "demo"}}},
        config={"configurable": {"user_id": "someone-else"}},
    )

    assert result["holdings"] == []


def test_portfolio_without_a_user_raises() -> None:
    with pytest.raises(MissingUserError):
        get_user_portfolio.invoke({}, config={"configurable": {}})


def _raise(error: Exception):
    def fail(*args, **kwargs):
        raise error

    return fail


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (market_data.TickerNotFoundError("no quote for 'ZZZZ'"), "not_found"),
        (market_data.MarketDataError("Finnhub rate limit reached"), "unavailable"),
    ],
)
def test_market_data_failures_are_returned_not_raised(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: str
) -> None:
    monkeypatch.setattr(market_data, "get_live_market_data", _raise(error))

    result = get_live_market_data.invoke({"ticker": "zzzz"})

    assert result == {"status": status, "ticker": "ZZZZ", "message": str(error)}


def test_10k_arguments_are_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake(ticker, question, section, fiscal_year, k, *, user_id):
        calls.append((ticker, question, section, fiscal_year, k, user_id))
        return TenKSearchResult(
            status="ok",
            ticker="AAPL",
            fiscal_year=2024,
            section=section,
            available_years=[2025, 2024],
        )

    monkeypatch.setattr(filings, "query_company_10k", fake)

    result = query_company_10k.invoke(
        {
            "ticker": "AAPL",
            "question": "risks",
            "section": "Item 1A",
            "fiscal_year": 2024,
            # Sent by the model: must not become the user.
            "user_id": "mallory",
        },
        config={"configurable": {"user_id": "alice"}},
    )

    assert calls == [("AAPL", "risks", "Item 1A", 2024, 5, "alice")]
    assert result["status"] == "ok"
    assert result["available_years"] == [2025, 2024]


def test_10k_section_outside_the_enum_is_rejected_before_searching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        filings, "query_company_10k", _raise(AssertionError("searched"))
    )

    with pytest.raises(ValueError, match="Item 1A"):
        query_company_10k.invoke(
            {"ticker": "AAPL", "question": "risks", "section": "Risk Factors"}
        )


def test_10k_bad_k_comes_back_as_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        filings,
        "query_company_10k",
        _raise(ValueError("k must be between 1 and 20; got 50")),
    )

    result = query_company_10k.invoke({"ticker": "AAPL", "question": "risks", "k": 50})

    assert result == {
        "status": "invalid_arguments",
        "message": "k must be between 1 and 20; got 50",
    }


# --- financial history: hand-calculated figures (6.2.6) ----------------------


def _figure(name: str, unit: str, values: dict[int, float]) -> Figure:
    return Figure(name=name, unit=unit, concept=None, label=None, values=values)


HISTORY = FinancialHistory(
    ticker="TEST",
    accession="0000000000-25-000001",
    fiscal_years=[2025, 2024, 2023],
    fiscal_year_ends={
        2025: dt.date(2025, 9, 27),
        2024: dt.date(2024, 9, 28),
        2023: dt.date(2023, 9, 30),
    },
    figures={
        "revenue": _figure("revenue", "USD", {2023: 100.0, 2024: 110.0, 2025: 121.0}),
        "net_income": _figure("net_income", "USD", {2023: -10.0, 2024: 5.0, 2025: 10.0}),
        "eps_diluted": _figure("eps_diluted", "USD/share", {2023: 1.0, 2024: 1.25, 2025: 1.5}),
        "gross_margin": _figure("gross_margin", "ratio", {2023: 0.40, 2024: 0.42, 2025: 0.45}),
    },
    debt_components=[],
    revenue_breakdowns=[],
    unavailable={"operating_margin": "no operating income reported"},
)  # fmt: skip


@pytest.fixture
def financial_result(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(financials, "get_financial_history", lambda t: HISTORY)
    return get_financial_history.invoke({"ticker": "test"})


def _changes(result: dict, figure: str) -> list[tuple]:
    return [
        (c["to_year"], c["change"], c["growth"], c["note"])
        for c in result["trends"]["year_over_year"][figure]
    ]


def test_financial_history_citation_names_the_newest_year(financial_result) -> None:
    assert financial_result["status"] == "ok"
    assert financial_result["citation"] == "[TEST 10-K FY2025, Item 8]"
    assert FILING_CITATION.fullmatch(financial_result["citation"])
    assert financial_result["figures"]["revenue"]["values"]["2025"] == 121.0


def test_revenue_growth_by_hand(financial_result) -> None:
    # 110 / 100 - 1 and 121 / 110 - 1; newest first.
    assert _changes(financial_result, "revenue") == [
        (2025, pytest.approx(11.0), pytest.approx(0.10), None),
        (2024, pytest.approx(10.0), pytest.approx(0.10), None),
    ]
    span = financial_result["trends"]["span"]["revenue"]
    # 121 / 100 - 1 over two years; (1.21) ** (1/2) - 1.
    assert (span["from_year"], span["to_year"]) == (2023, 2025)
    assert span["growth"] == pytest.approx(0.21)
    assert span["cagr"] == pytest.approx(0.10)


def test_growth_from_a_loss_is_not_meaningful(financial_result) -> None:
    # -10 to 5: change +15, no growth rate; 5 to 10: +100%.
    assert _changes(financial_result, "net_income") == [
        (2025, pytest.approx(5.0), pytest.approx(1.0), None),
        (2024, pytest.approx(15.0), None, NOT_MEANINGFUL),
    ]
    span = financial_result["trends"]["span"]["net_income"]
    assert span["change"] == pytest.approx(20.0)
    assert (span["growth"], span["note"]) == (None, NOT_MEANINGFUL)


def test_eps_growth_and_cagr_by_hand(financial_result) -> None:
    assert [g for _, _, g, _ in _changes(financial_result, "eps_diluted")] == [
        pytest.approx(0.20),  # 1.5 / 1.25 - 1
        pytest.approx(0.25),  # 1.25 / 1.0 - 1
    ]
    span = financial_result["trends"]["span"]["eps_diluted"]
    assert span["growth"] == pytest.approx(0.5)
    assert span["cagr"] == pytest.approx(0.224744871391589)  # 1.5 ** 0.5 - 1


def test_margin_changes_in_percentage_points(financial_result) -> None:
    margin = financial_result["trends"]["year_over_year"]["gross_margin"]
    assert [(c["change"], c["unit"]) for c in margin] == [
        (pytest.approx(3.0), "percentage points"),  # 45% - 42%
        (pytest.approx(2.0), "percentage points"),  # 42% - 40%
    ]
    assert financial_result["trends"]["span"]["gross_margin"][
        "change"
    ] == pytest.approx(5.0)
    assert "operating_margin" in financial_result["trends"]["unavailable"]


def test_financial_history_without_a_10k_is_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        financials,
        "get_financial_history",
        _raise(LookupError("no 10-K filings found")),
    )
    result = get_financial_history.invoke({"ticker": "spy"})
    assert result == {
        "status": "not_found",
        "ticker": "SPY",
        "message": "no 10-K filings found",
    }


# --- price history: hand-calculated figures (6.2.6) --------------------------

_D = dt.date
_BARS = [
    prices.Bar(_D(2025, 6, 13), 101, 99, 100),  # 1y base: 2025-06-15 is a Sunday
    prices.Bar(_D(2025, 12, 1), 81, 79, 80),
    prices.Bar(_D(2026, 6, 15), 121, 119, 120),
]


@pytest.fixture
def price_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real computation over three bars, with Alpaca and EDGAR replaced."""
    monkeypatch.setattr(prices, "_bars", lambda symbol, adjustment, start, end: _BARS)
    monkeypatch.setattr(
        prices,
        "_fiscal_year_ends",
        lambda s: ([_D(2025, 6, 13), _D(2026, 6, 15)], None),
    )
    monkeypatch.setattr(prices, "_today", lambda: _D(2026, 6, 16))
    prices._cache.clear()
    yield
    prices._cache.clear()


def test_price_history_by_hand(price_bars) -> None:
    result = get_price_history.invoke({"ticker": "test"})
    assert result["status"] == "ok"
    assert result["citation"] == "[Alpaca prices, to 2026-06-15]"
    assert PRICE_CITATION.fullmatch(result["citation"])
    assert result["period"] == "1y"
    assert result["total_return"] == pytest.approx(0.20)  # 120 / 100 - 1
    assert result["max_drawdown"] == pytest.approx(-0.20)  # 80 / 100 - 1
    # Log returns ln(0.8), ln(1.5): sample sd 0.444493445790, times sqrt(252).
    assert result["annualised_volatility"] == pytest.approx(7.056114701753)
    assert (result["week_52_high"], result["week_52_low"]) == (121, 79)
    [fy] = result["fiscal_year_returns"]
    assert (fy["fiscal_year"], fy["total_return"]) == (2026, pytest.approx(0.20))


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (market_data.TickerNotFoundError("no price history for 'ZZZZQ'"), "not_found"),
        (prices.PriceHistoryError("Alpaca rate limit reached"), "unavailable"),
    ],
)
def test_price_history_failures_are_returned_not_raised(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: str
) -> None:
    monkeypatch.setattr(prices, "get_price_history", _raise(error))
    result = get_price_history.invoke({"ticker": "zzzzq", "period": "5y"})
    assert result == {"status": status, "ticker": "ZZZZQ", "message": str(error)}


def test_price_period_outside_the_enum_is_rejected() -> None:
    with pytest.raises(ValueError, match="1y"):
        get_price_history.invoke({"ticker": "AAPL", "period": "10y"})
