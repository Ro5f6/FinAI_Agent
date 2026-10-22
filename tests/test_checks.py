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


# --- citations (1.5.5) ------------------------------------------------------------

from finai.agent.verification.checks import check_citations

DOCS = [
    {"ticker": "NVDA", "fiscal_year": 2026, "section": "Item 1A", "text": "..."},
    {"ticker": "NVDA", "fiscal_year": 2026, "section": "Item 7", "text": "..."},
]
CITED_RESULTS = [
    {**QUOTE, "output": {**QUOTE["output"], "quote_time": "2026-10-22T20:00:00Z"}},
    {
        "tool": "get_financial_history",
        "status": "success",
        "output": {"citation": "[NVDA 10-K FY2026, Item 8]"},
    },
    {
        "tool": "get_price_history",
        "status": "success",
        "output": {"end_date": "2026-10-21"},
    },
    {"tool": "get_user_portfolio", "status": "success", "output": {"holdings": []}},
]


def _cited(draft: str, results=CITED_RESULTS, docs=DOCS) -> list[str]:
    return [i.claim for i in check_citations(draft, results, docs)]


def test_every_citation_form_passes_when_its_source_was_returned() -> None:
    draft = (
        "Risks [NVDA 10-K FY2026, Item 1A]. Results [NVDA 10-K FY2026, Item 7]. "
        "Revenue [NVDA 10-K FY2026, Item 8]. Price [Finnhub quote, 2026-10-22 20:00 UTC]. "
        "Return [Alpaca prices, to 2026-10-21]. Holdings [portfolio]. "
        "Repeated [NVDA 10-K FY2026, Item 1A]."
    )
    assert _cited(draft) == []


@pytest.mark.parametrize(
    "citation",
    [
        "[NVDA 10-K FY2024, Item 1A]",  # a year not retrieved
        "[NVDA 10-K FY2026, Item 7A]",  # a section not retrieved
        "[AMD 10-K FY2026, Item 1A]",  # a company not retrieved
        "[Finnhub quote, 2026-10-22 19:00 UTC]",  # a quote time not returned
        "[Alpaca prices, to 2026-10-14]",  # a price history not returned
    ],
)
def test_a_citation_to_a_source_not_returned_is_flagged(citation: str) -> None:
    assert _cited(f"A claim {citation}.") == [citation]


def test_portfolio_cited_without_reading_it_is_flagged() -> None:
    assert _cited("You hold 10 AAPL [portfolio].", results=[]) == ["[portfolio]"]


@pytest.mark.parametrize(
    "citation",
    [
        "[NVDA 10-K 2026, Item 1A]",
        "[NVDA 10-K FY2026, Risk Factors]",
        "[Finnhub, today]",
    ],
)
def test_a_loose_citation_form_is_flagged(citation: str) -> None:
    [issue] = check_citations(f"A claim {citation}.", CITED_RESULTS, DOCS)
    assert issue.claim == citation
    assert "citation form" in issue.reason


def test_links_and_other_brackets_are_not_citations() -> None:
    assert _cited("See [the filing](https://sec.gov/x) and [note 3].") == []


def test_a_failed_tool_call_supports_no_citation() -> None:
    failed = [
        {"tool": "get_live_market_data", "status": "error", "output": "bad ticker"}
    ]
    assert _cited("[Finnhub quote, 2026-10-22 20:00 UTC]", results=failed) == [
        "[Finnhub quote, 2026-10-22 20:00 UTC]"
    ]


# --- portfolio math (1.5.6) ---------------------------------------------------------

from finai.agent.verification.checks import check_portfolio_math

VALUATION = {
    "tool": "get_portfolio_value",
    "status": "success",
    "output": {
        "positions": [
            {"ticker": "AAPL", "shares": "10", "avg_cost": "175.20", "price": "258.45",
             "quote_time": "2026-10-22T20:00:00Z", "market_value": "2584.50"},
            {"ticker": "NVDA", "shares": "25", "avg_cost": "98.50", "price": "212.21",
             "quote_time": "2026-10-22T20:00:00Z", "market_value": "5305.25"},
        ],
        "total_market_value": "7889.75",
    },
}  # fmt: skip


def _math(draft: str, results=(VALUATION,)) -> list[str]:
    return [i.claim for i in check_portfolio_math(draft, results)]


def test_correct_portfolio_amounts_pass() -> None:
    draft = (
        "Your holdings are worth $7,889.75: AAPL $2,584.50 and NVDA $5,305.25, "
        "against a cost of $4,214.50, a gain of $3,675.25."
    )
    assert _math(draft) == []


@pytest.mark.parametrize("wrong", ["$7,889.76", "$7,898.75", "$5,305.52", "$3,675.00"])
def test_an_amount_off_by_a_cent_or_more_is_flagged(wrong: str) -> None:
    [issue] = check_portfolio_math(f"Your holdings are worth {wrong}.", [VALUATION])
    assert issue.claim == wrong
    assert "recomputed in Python" in issue.reason


def test_a_wrong_figure_from_the_tool_is_caught_by_the_recomputation() -> None:
    """The tool said 7,889.57; shares x price says 7,889.75."""
    bugged = {
        **VALUATION,
        "output": {**VALUATION["output"], "total_market_value": "7889.57"},
    }
    assert _math("Your holdings are worth $7,889.57.", [bugged]) == ["$7,889.57"]


def test_amounts_unrelated_to_the_portfolio_are_left_to_the_number_check() -> None:
    assert _math("NVDA's charge was $4.5 billion; the stock is at $212.21.") == []
    assert _math("Your holdings are worth $7,889.76.", results=[]) == []


def test_valuation_quote_times_support_finnhub_citations() -> None:
    assert (
        check_citations(
            "[Finnhub quote, 2026-10-22 20:00 UTC] [portfolio]", [VALUATION], []
        )
        == []
    )
