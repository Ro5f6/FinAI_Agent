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


class _Element:
    def __init__(self, label: str) -> None:
        self.labels = {"http://www.xbrl.org/2003/role/label": label}


class _XBRL:
    """Stand-in for edgartools' XBRL: ``query()`` over a fact table, the
    balance sheet's debt rows, dimensional facts and member labels."""

    def __init__(
        self,
        facts: list[dict],
        balance_rows: list[dict] | None = None,
        dimension_facts: list[dict] | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self._facts = pd.DataFrame(facts)
        dims = pd.DataFrame(dimension_facts or [], columns=_DIM_COLUMNS)
        self.facts = type(
            "Facts", (), {"get_facts_with_dimensions": lambda _self: dims}
        )()
        self.element_catalog = {k: _Element(v) for k, v in (labels or {}).items()}
        rows = balance_rows or []
        columns = [
            "concept",
            "label",
            "standard_concept",
            "dimension",
            ENDS[2025],
            ENDS[2024],
        ]
        sheet = pd.DataFrame(rows, columns=columns)
        self.statements = type(
            "S", (), {"balance_sheet": lambda _self: _Statement(sheet)}
        )()

    def query(self) -> _Query:
        return _Query(self._facts)


_DIM_COLUMNS = [
    "concept",
    "numeric_value",
    "period_start",
    "period_end",
    "dim_srt_ProductOrServiceAxis",
    "dim_srt_StatementGeographicalAxis",
    "dim_us-gaap_StatementBusinessSegmentsAxis",
    "dim_srt_ConsolidationItemsAxis",
    "dim_us-gaap_EquityMethodInvestmentNonconsolidatedInvesteeAxis",
]


def _dim_fact(concept: str, value: float, year: int, **dims: str) -> dict:
    """One dimensional fact; ``dims`` maps dim_* column names to members."""
    row = {c: None for c in _DIM_COLUMNS}
    row.update(
        concept=f"us-gaap:{concept}",
        numeric_value=value,
        period_start=STARTS[year],
        period_end=ENDS[year],
    )
    row.update({f"dim_{k}": v for k, v in dims.items()})
    return row


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


# --- revenue breakdowns --------------------------------------------------------

REV = "RevenueFromContractWithCustomerExcludingAssessedTax"
SEG = {"srt_ConsolidationItemsAxis": "us-gaap:OperatingSegmentsMember"}


def _apple_with(dimension_facts: list[dict], labels: dict | None = None) -> _XBRL:
    base = _apple_like()
    return _XBRL(base._facts.to_dict("records"), None, dimension_facts, labels)


def _by_axis(h, axis: str):
    return next(b for b in h.revenue_breakdowns if b.axis == axis)


def _segments(years=(2025, 2024, 2023)) -> list[dict]:
    split = {
        2025: (178_353e6, 237_808e6),
        2024: (167_045e6, 223_990e6),
        2023: (162_560e6, 220_725e6),
    }
    return [
        f
        for y in years
        for f in (
            _dim_fact(
                REV,
                split[y][0],
                y,
                **SEG,
                **{
                    "us-gaap_StatementBusinessSegmentsAxis": "aapl:AmericasSegmentMember"
                },
            ),
            _dim_fact(
                REV,
                split[y][1],
                y,
                **SEG,
                **{"us-gaap_StatementBusinessSegmentsAxis": "aapl:RestOfWorldMember"},
            ),
        )
    ]


def test_segments_ignore_the_qualifier_and_add_up() -> None:
    labels = {"aapl_AmericasSegmentMember": "Americas Segment [Member]"}

    h = financial_history(_Filing(_apple_with(_segments(), labels)), "AAPL")
    seg = _by_axis(h, "segment")

    assert [m.label for m in seg.members] == ["Americas Segment", "Rest Of World"]
    assert seg.members[0].values[2025] == 178_353e6
    assert seg.adds_up_to_total


def test_nested_members_are_flagged_as_not_adding_up() -> None:
    """AAPL lists "Products" and the iPhone inside it on the same axis."""
    facts = [
        _dim_fact(REV, v, y, srt_ProductOrServiceAxis=m)
        for y in (2025, 2024, 2023)
        for m, v in (
            ("us-gaap:ProductMember", 307_003e6),
            ("aapl:IPhoneMember", 209_586e6),
            ("us-gaap:ServiceMember", 109_158e6),
        )
    ]

    product = _by_axis(
        financial_history(_Filing(_apple_with(facts)), "AAPL"), "product"
    )

    assert not product.adds_up_to_total
    assert {m.label for m in product.members} == {"Product", "IPhone", "Service"}


def test_two_axis_facts_are_left_out() -> None:
    """KO's concentrate x United States would double-count with the totals."""
    facts = _segments() + [
        _dim_fact(
            REV,
            9e9,
            2025,
            srt_ProductOrServiceAxis="ko:ConcentrateOperationsMember",
            srt_StatementGeographicalAxis="country:US",
        ),
    ]

    h = financial_history(_Filing(_apple_with(facts)), "AAPL")

    assert [b.axis for b in h.revenue_breakdowns] == ["segment"]


def test_other_axes_are_ignored() -> None:
    facts = _segments() + [
        _dim_fact(
            REV,
            102.8e9,
            2025,
            **{
                "us-gaap_EquityMethodInvestmentNonconsolidatedInvesteeAxis": "x:Investee"
            },
        ),
    ]

    assert [
        b.axis
        for b in financial_history(
            _Filing(_apple_with(facts)), "AAPL"
        ).revenue_breakdowns
    ] == ["segment"]


def test_country_members_keep_their_codes() -> None:
    facts = [
        _dim_fact(REV, v, y, srt_StatementGeographicalAxis=m)
        for y in (2025, 2024, 2023)
        for m, v in (
            ("country:US", 151_790e6),
            ("aapl:OtherCountriesMember", 264_371e6),
        )
    ]

    geo = _by_axis(financial_history(_Filing(_apple_with(facts)), "AAPL"), "geography")

    assert [m.label for m in geo.members] == ["US", "Other Countries"]


def test_conflicting_member_is_dropped_with_a_reason() -> None:
    """XOM reports some members twice with different values for one year."""
    facts = _segments() + [
        _dim_fact(
            REV,
            1.0e9,
            2025,
            **SEG,
            **{"us-gaap_StatementBusinessSegmentsAxis": "aapl:AmericasSegmentMember"},
        ),
    ]

    seg = _by_axis(financial_history(_Filing(_apple_with(facts)), "AAPL"), "segment")

    assert [m.label for m in seg.members] == ["Rest Of World"]
    assert seg.dropped == {"Americas Segment": "conflicting values for one year"}
    assert not seg.adds_up_to_total


def test_missing_year_breaks_adds_up() -> None:
    seg = _by_axis(
        financial_history(_Filing(_apple_with(_segments(years=(2025, 2024)))), "AAPL"),
        "segment",
    )

    assert seg.members[0].values[2023] is None
    assert not seg.adds_up_to_total


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


@pytest.mark.network
def test_live_breakdowns() -> None:
    aapl = financials.get_financial_history("AAPL")
    product = next(b for b in aapl.revenue_breakdowns if b.axis == "product")
    iphone = next(m for m in product.members if m.name == "aapl:IPhoneMember")
    assert iphone.values[2025] == 209_586e6
    assert not product.adds_up_to_total  # "Product" and iPhone nest

    wmt = financials.get_financial_history("WMT")
    assert next(
        b for b in wmt.revenue_breakdowns if b.axis == "segment"
    ).adds_up_to_total
