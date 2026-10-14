"""Tests for the financial-history tool (1.3.22).

Offline tests feed small fact tables to a stand-in XBRL object, each
reproducing a trap seen on real filings; ``-m network`` checks real figures.
"""

import datetime as dt

import pandas as pd
import pytest

from finai.tools import financials
from finai.tools.financials import financial_history

ENDS = {2025: "2025-09-27", 2024: "2024-09-28", 2023: "2023-09-30"}
STARTS = {2025: "2024-09-29", 2024: "2023-10-01", 2023: "2022-09-25"}


def _annual(
    concept: str,
    values: dict[int, float],
    label: str = "",
    dimensioned=False,
    fiscal_year_field=None,
) -> list[dict]:
    return [
        {
            "concept": f"us-gaap:{concept}",
            "label": label or concept,
            "numeric_value": v,
            "period_start": STARTS[y],
            "period_end": ENDS[y],
            "period_instant": None,
            "is_dimensioned": dimensioned,
            "fiscal_year": fiscal_year_field if fiscal_year_field is not None else y,
        }
        for y, v in values.items()
    ]


def _instant(concept: str, values: dict[str, float]) -> list[dict]:
    return [
        {
            "concept": f"us-gaap:{concept}",
            "label": concept,
            "numeric_value": v,
            "period_start": None,
            "period_end": None,
            "period_instant": day,
            "is_dimensioned": False,
            "fiscal_year": 2025,
        }
        for day, v in values.items()
    ]


class _Query:
    def __init__(self, facts: pd.DataFrame) -> None:
        self.facts = facts

    def by_concept(self, pattern: str, exact: bool = False) -> "_Query":
        return _Query(self.facts[self.facts["concept"] == pattern])

    def to_dataframe(self) -> pd.DataFrame:
        return self.facts.copy()


class _Statement:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self.df


class _XBRL:
    def __init__(
        self, facts: list[dict], balance_rows: list[dict] | None = None
    ) -> None:
        self.facts = pd.DataFrame(facts)
        rows = balance_rows or []
        self.statements = type(
            "S",
            (),
            {
                "balance_sheet": lambda _self: _Statement(
                    pd.DataFrame(
                        rows,
                        columns=[
                            "concept",
                            "label",
                            "standard_concept",
                            "dimension",
                            ENDS[2025],
                            ENDS[2024],
                        ],
                    )
                )
            },
        )()

    def query(self) -> _Query:
        return _Query(self.facts)


class _Filing:
    form = "10-K"
    accession_no = "0000320193-25-000079"
    filing_date = dt.date(2025, 10, 31)
    period_of_report = "2025-09-27"

    def __init__(self, xbrl: _XBRL) -> None:
        self._xbrl = xbrl

    def xbrl(self) -> _XBRL:
        return self._xbrl


def _debt_row(standard: str, label: str, now: float, before: float) -> dict:
    return {
        "concept": f"us-gaap_{standard}X",
        "label": label,
        "standard_concept": standard,
        "dimension": False,
        ENDS[2025]: now,
        ENDS[2024]: before,
    }


def _apple_like(**extra) -> _XBRL:
    facts = (
        _annual(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            {2025: 416_161e6, 2024: 391_035e6, 2023: 383_285e6},
            "Net sales",
        )
        # Breakdown facts share the concept and periods: they must be ignored.
        + _annual(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            {2025: 209_586e6, 2024: 201_183e6, 2023: 200_583e6},
            "iPhone",
            dimensioned=True,
        )
        + _annual("GrossProfit", {2025: 195_201e6, 2024: 180_683e6, 2023: 169_148e6})
        + _annual(
            "OperatingIncomeLoss", {2025: 133_050e6, 2024: 123_216e6, 2023: 114_301e6}
        )
        + _annual("NetIncomeLoss", {2025: 112_010e6, 2024: 93_736e6, 2023: 96_995e6})
        + _annual("EarningsPerShareDiluted", {2025: 7.46, 2024: 6.08, 2023: 6.13})
        + _instant(
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            {
                "2025-09-27": 35_934e6,
                "2024-09-28": 29_943e6,
                "2023-09-30": 30_737e6,
                "2022-09-24": 24_977e6,
            },
        )
        + extra.get("facts", [])
    )
    debt = extra.get(
        "debt",
        [
            _debt_row("ShortTermDebt", "Commercial paper", 7_979e6, 9_967e6),
            _debt_row("CurrentPortionOfLongTermDebt", "Term debt", 12_350e6, 10_912e6),
            _debt_row("LongTermDebt", "Term debt", 78_328e6, 85_750e6),
        ],
    )
    return _XBRL(facts, debt)


def test_headline_figures_for_three_years() -> None:
    h = financial_history(_Filing(_apple_like()), "aapl")

    assert h.ticker == "AAPL"
    assert h.fiscal_years == [2025, 2024, 2023]
    assert h.figures["revenue"].values == {
        2025: 416_161e6,
        2024: 391_035e6,
        2023: 383_285e6,
    }
    assert h.figures["eps_diluted"].values[2023] == 6.13
    assert h.figures["net_income"].concept == "NetIncomeLoss"


def test_breakdown_facts_are_not_mistaken_for_the_total() -> None:
    """exclude_dimensions() kept iPhone's $209.6B next to the $416.2B total."""
    h = financial_history(_Filing(_apple_like()), "AAPL")

    assert h.figures["revenue"].values[2025] == 416_161e6
    assert h.figures["revenue"].label == "Net sales"


def test_fiscal_year_comes_from_the_period_end_not_the_fact_field() -> None:
    """edgartools labels comparative periods with the filing's year."""
    facts = _annual("Revenues", {2024: 1.0, 2023: 2.0}, fiscal_year_field=2025)
    ends = financials.fiscal_year_ends(_XBRL(facts))

    assert list(ends) == [2024, 2023]
    assert ends[2024] == dt.date(2024, 9, 28)


def test_total_revenue_concept_wins_over_components() -> None:
    """JPM: RevenuesNetOfInterestExpense ($182B), not a $27B revenue line."""
    xbrl = _apple_like(
        facts=_annual(
            "RevenuesNetOfInterestExpense",
            {2025: 182_447e6, 2024: 177_556e6, 2023: 158_104e6},
        )
    )

    h = financial_history(_Filing(xbrl), "JPM")

    assert h.figures["revenue"].concept == "RevenuesNetOfInterestExpense"
    assert h.figures["revenue"].values[2025] == 182_447e6


def test_quarterly_facts_are_ignored() -> None:
    q4 = [
        {
            "concept": "us-gaap:Revenues",
            "label": "Revenues",
            "numeric_value": 1e9,
            "period_start": "2025-06-29",
            "period_end": "2025-09-27",
            "period_instant": None,
            "is_dimensioned": False,
            "fiscal_year": 2025,
        }
    ]
    facts = _annual("Revenues", {2025: 4e9, 2024: 3e9, 2023: 2e9}) + q4

    h = financial_history(_Filing(_XBRL(facts)), "X")

    assert h.figures["revenue"].values[2025] == 4e9


def test_conflicting_values_for_one_period_raise() -> None:
    facts = _annual("Revenues", {2025: 4e9, 2024: 3e9, 2023: 2e9}) + _annual(
        "Revenues", {2025: 5e9}
    )

    with pytest.raises(ValueError, match="conflicting"):
        financial_history(_Filing(_XBRL(facts)), "X")


def test_third_year_cash_is_the_cash_flow_opening_balance() -> None:
    h = financial_history(_Filing(_apple_like()), "AAPL")

    assert h.figures["cash"].values == {2025: 35_934e6, 2024: 29_943e6, 2023: 30_737e6}


def test_debt_has_two_years_and_says_why_not_three() -> None:
    h = financial_history(_Filing(_apple_like()), "AAPL")

    assert h.figures["total_debt"].values == {
        2025: 7_979e6 + 12_350e6 + 78_328e6,
        2024: 9_967e6 + 10_912e6 + 85_750e6,
        2023: None,
    }
    assert [c.label for c in h.debt_components] == [
        "Commercial paper",
        "Term debt",
        "Term debt",
    ]
    assert "2023" in h.unavailable["total_debt_years"]


def test_margins_are_computed_from_reported_figures() -> None:
    h = financial_history(_Filing(_apple_like()), "AAPL")

    assert h.figures["gross_margin"].values[2025] == pytest.approx(195_201 / 416_161)
    assert h.figures["operating_margin"].values[2024] == pytest.approx(
        123_216 / 391_035
    )
    assert h.figures["gross_margin"].unit == "ratio"


def test_no_gross_margin_without_reported_gross_profit() -> None:
    """Banks, oil majors, WMT and F report no gross profit: no improvised margin."""
    facts = _annual("Revenues", {2025: 4e9, 2024: 3e9, 2023: 2e9}) + _annual(
        "CostOfRevenue", {2025: 1e9, 2024: 1e9, 2023: 1e9}
    )

    h = financial_history(_Filing(_XBRL(facts)), "X")

    assert "gross_margin" not in h.figures
    assert h.unavailable["gross_margin"] == "the company reports no gross profit"


def test_net_income_falls_back_to_common_stockholders() -> None:
    """Ford tags net income only as attributable to common stockholders."""
    facts = _annual("Revenues", {2025: 4e9, 2024: 3e9, 2023: 2e9}) + _annual(
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        {2025: -8.18e9, 2024: 5.88e9, 2023: 4.35e9},
    )

    h = financial_history(_Filing(_XBRL(facts)), "F")

    assert h.figures["net_income"].values[2025] == -8.18e9
    assert (
        h.figures["net_income"].concept
        == "NetIncomeLossAvailableToCommonStockholdersBasic"
    )


@pytest.mark.network
@pytest.mark.parametrize(
    ("ticker", "year", "revenue", "eps"),
    [
        ("AAPL", 2025, 416_161e6, 7.46),
        ("NVDA", 2026, 215_938e6, 4.90),
        ("JPM", 2025, 182_447e6, 20.02),
        ("WMT", 2026, 713_163e6, 2.73),
        ("F", 2025, 187_267e6, -2.06),
    ],
)
def test_live_figures_match_the_filings(
    ticker: str, year: int, revenue: float, eps: float
) -> None:
    h = financials.get_financial_history(ticker)

    assert h.fiscal_years[0] == year
    assert h.figures["revenue"].values[year] == pytest.approx(revenue, rel=1e-3)
    assert h.figures["eps_diluted"].values[year] == pytest.approx(eps)
    assert h.figures["total_debt"].values[h.fiscal_years[2]] is None
