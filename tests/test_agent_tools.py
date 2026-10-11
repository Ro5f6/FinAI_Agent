"""Tests for the LangChain tool wrappers: what the model sees, and that
user_id comes only from the run's config."""

import typing

import pytest

from finai.ingestion.parse import TARGET_ITEMS
from finai.tools import agent_tools, filings, market_data
from finai.tools.agent_tools import (
    TOOLS,
    MissingUserError,
    get_live_market_data,
    get_user_portfolio,
    query_company_10k,
)
from finai.tools.filings import TenKSearchResult


def _schema(tool) -> dict:
    return tool.tool_call_schema.model_json_schema()


def test_the_three_tools_are_registered() -> None:
    assert [t.name for t in TOOLS] == [
        "get_live_market_data",
        "query_company_10k",
        "get_user_portfolio",
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


def test_required_arguments() -> None:
    assert _schema(get_live_market_data)["required"] == ["ticker"]
    assert _schema(query_company_10k)["required"] == ["ticker", "question"]
    assert _schema(get_user_portfolio).get("required", []) == []


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

    def fake(ticker, question, section, fiscal_year, k):
        calls.append((ticker, question, section, fiscal_year, k))
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
        }
    )

    assert calls == [("AAPL", "risks", "Item 1A", 2024, 5)]
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
