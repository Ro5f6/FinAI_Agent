"""Tests for the deterministic checks on a draft answer (1.5).

The fixtures copy the shape and values of real tool results from the 1.3.20
run (NVDA, 2026-10-22); docs/CONTEXT.md has the measurement on live runs.
"""

import pytest

from finai.agent.verification.checks import check_numbers, figures, source_values

QUOTE = {
    "tool": "get_live_market_data",
    "tool_call_id": "c1",
    "args": {"ticker": "NVDA"},
    "status": "success",
    "output": {
        "status": "ok",
        "ticker": "NVDA",
        "price": 212.21,
        "change_pct": -4.397,
        "market_cap": 5247534028858.0,
        "week_52_high": 243.37,
        "week_52_low": 164.27,
        "pe_ttm": 26.825,
        "pe_annual": 43.327,
        "eps_ttm": 7.9108,
        "eps_annual": 4.8979,
        "eps_ttm_period_end": "2026-07-26",
        "quote_time": "2026-10-22T18:01:00Z",
    },
}
SEARCH = {
    "tool": "query_company_10k",
    "tool_call_id": "c2",
    "args": {"ticker": "NVDA", "question": "risks"},
    "status": "success",
    "output": {
        "status": "ok",
        "ticker": "NVDA",
        "fiscal_year": 2026,
        "passages": [
            {
                "section": "Item 1A",
                "text": "We recorded a $4.5 billion charge in the first quarter of fiscal "
                "year 2026. One direct customer represented 22% of total revenue and "
                "another 14%. Revenue from outside the United States was 31%. Net sales "
                "by market ($ in millions): Data Center 115,186.",
            }
        ],
    },
}
HISTORY = {
    "tool": "get_financial_history",
    "tool_call_id": "c3",
    "args": {"ticker": "NVDA"},
    "status": "success",
    "output": {"trends": {"span": {"revenue": {"growth": 0.6547, "cagr": 0.0642551}}}},
}
RESULTS = [QUOTE, SEARCH, HISTORY]


def _flagged(draft: str, results=RESULTS, question: str = "") -> list[str]:
    return [i.claim for i in check_numbers(draft, results, question)]


# --- which numbers are figures ----------------------------------------------------


def test_figures_are_money_percentages_scaled_decimals_and_multiples() -> None:
    text = (
        "NVDA is at $212.21, down 4.40%; market cap about $5.25 trillion; "
        "P/E 26.8x or about 27 times; a $4.5 billion charge; 115,186; 43.3."
    )
    assert [f.text for f in figures(text)] == [
        "$212.21",
        "4.40%",
        "$5.25 trillion",
        "26.8x",
        "27 times",
        "$4.5 billion",
        "115,186",
        "43.3",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "A100, H100, H20 and H200s",
        "Q1 FY2026 10-K",
        "the 52-week range",
        "Country Groups D:1, D:4 and D:5",
        "Item 1A and Item 7",
        "period ending 2026-07-26 at 18:01 UTC",
        "the S&P 500 and two risks",
        "[NVDA 10-K FY2026, Item 1A] [Finnhub quote, 2026-10-22 18:01 UTC]",
    ],
)
def test_names_codes_dates_and_citations_are_not_figures(text: str) -> None:
    assert figures(text) == []


# --- matching ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "draft",
    [
        "NVDA is trading at $212.21.",  # exact
        "down 4.40% on the day",  # rounded; the sign is in words
        "about $5.25 trillion",  # 5,247,534,028,858 rounded and scaled
        "a P/E of about 26.8, or 27 times",  # 26.825 at one and zero decimals
        "EPS of $7.9108 and $4.90",  # exact, and 4.8979 rounded
        "a $4.5 billion charge",  # from passage text
        "22% and 14% of revenue",  # percentages in passage text
        "Data Center sales of $115.2 billion",  # a table value in millions
        "revenue grew 65.5%, 6.4% a year",  # fractions from the trends (D-036)
    ],
)
def test_supported_figures_pass(draft: str) -> None:
    assert _flagged(draft) == []


@pytest.mark.parametrize(
    ("draft", "claim"),
    [
        ("NVDA is trading at $221.21.", "$221.21"),  # digits swapped
        ("a P/E of 28.6", "28.6"),
        ("about $5.75 trillion", "$5.75 trillion"),
        ("a $5.4 billion charge", "$5.4 billion"),
        ("EPS of $7.19", "$7.19"),
        ("revenue grew 75.5%", "75.5%"),
    ],
)
def test_wrong_figures_are_flagged(draft: str, claim: str) -> None:
    assert _flagged(draft) == [claim]


def test_rounding_further_than_shown_is_not_allowed() -> None:
    """26.825 is "26.8" or "27"; "26.9" claims a precision the value lacks."""
    assert _flagged("a P/E of 26.9") == ["26.9"]


def test_a_figure_from_the_question_is_supported() -> None:
    assert _flagged("At a P/E of 30.5, ...", question="Is a P/E of 30.5 high?") == []


def test_the_reason_names_the_nearest_tool_value() -> None:
    [issue] = check_numbers("NVDA is trading at $221.21.", RESULTS)
    assert "221.21 is not in any tool result" in issue.reason
    assert "212.21" in issue.reason


def test_without_tool_results_every_figure_is_unsupported() -> None:
    assert _flagged("NVDA is at $212.21, a P/E of 26.8.", results=[]) == [
        "$212.21",
        "26.8",
    ]


def test_error_results_and_odd_values_are_tolerated() -> None:
    results = [
        {
            "tool": "query_company_10k",
            "status": "error",
            "output": "section must be one of",
        },
        {
            "tool": "x",
            "status": "success",
            "output": {"a": None, "b": True, "c": float("nan")},
        },
    ]
    assert source_values(r["output"] for r in results) == []


def test_known_gap_a_number_elsewhere_in_the_passages_passes() -> None:
    """The check confirms a figure exists in the results, not that it belongs
    to the claim: 14% is another customer's share, so "22%" -> "14%" passes.
    The LLM critic (1.5.8) reads the claim in context."""
    assert _flagged("One direct customer represented 14% of revenue.") == []
