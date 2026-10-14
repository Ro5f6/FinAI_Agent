"""Three fiscal years of headline figures from a 10-K's XBRL (1.3.22).

A 10-K carries three years of income and cash-flow figures and two balance
sheets, so this needs no request beyond the latest filing. Figures come from
the XBRL facts, by concept and exact period, not from statement rows or the
facts' own ``fiscal_year`` field. Verified against edgartools 5.60.0 on ten
filers (docs/CONTEXT.md):

* The facts' ``fiscal_year`` is wrong for comparative periods (AAPL's cash
  at 2024-09-28 is labelled 2025), so the fiscal year is the year of each
  period's end, the rule ingestion uses (D-008).
* Companies tag the same line with different concepts, and edgartools'
  ``standard_concept`` can pick a component: JPM's "Revenue" is principal
  transactions ($27B), its total is ``RevenuesNetOfInterestExpense``
  ($182B). Each figure therefore has an explicit concept priority list and
  records which concept and label it used.
* ``exclude_dimensions()`` doesn't exclude breakdowns (iPhone, regions), so
  totals are picked by ``is_dimensioned``; conflicting values for one period
  raise instead of picking one.
* Cash comes from the cash-flow statement's instants, which include the year
  before the earliest income year: the third year's cash is the opening
  balance the plan asked for.
* Breakdowns (revenue by product, geography, business segment) come from
  ``facts.get_facts_with_dimensions()``, which gives every axis of a fact;
  the statement view shows one axis per row and mislabels some (XOM revenue
  members labelled "Pension Benefits"). A breakdown member is a fact with
  exactly one of those three axes, ignoring the ``ConsolidationItemsAxis =
  OperatingSegmentsMember`` qualifier segment facts carry. Facts with two
  axes (KO concentrate x US) are left out: mixing them double-counts. Many
  axes hold nested or overlapping members (AAPL "Products" and iPhone, F's
  two segment groupings), so each breakdown says whether its members add up
  to the total; when they don't, they must not be summed.
* Debt is the sum of the debt lines presented on the balance sheet; it has
  only two years, so the third year's debt is reported as unavailable.

Margins are computed here, only from reported figures: no gross margin is
improvised from revenue minus some cost line when a company reports no gross
profit (banks, oil majors, WMT, F).
"""

import datetime as dt
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from finai.ingestion.fetch import filing_meta, get_latest_10k

# A fiscal year's duration in days; excludes quarters and 52/53-week oddities
# outside this band.
_YEAR_DAYS = (350, 380)
YEARS = 3

REVENUE = (
    "RevenuesNetOfInterestExpense",  # banks: total net revenue
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
GROSS_PROFIT = ("GrossProfit",)
OPERATING_INCOME = ("OperatingIncomeLoss",)
NET_INCOME = (
    "NetIncomeLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",  # F
    "ProfitLoss",
)
EPS_DILUTED = ("EarningsPerShareDiluted",)
CASH = (
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "CashAndCashEquivalentsAtCarryingValue",
)
# edgartools' standard concepts for balance-sheet debt lines.
DEBT_LINES = ("ShortTermDebt", "CurrentPortionOfLongTermDebt", "LongTermDebt")


# The axes reported as breakdowns, by the name used in the result.
BREAKDOWN_AXES = {
    "srt:ProductOrServiceAxis": "product",
    "srt:StatementGeographicalAxis": "geography",
    "us-gaap:StatementBusinessSegmentsAxis": "segment",
}
# Segment facts carry this as a second dimension; it marks them as segment
# figures and is not a breakdown of its own.
_SEGMENT_QUALIFIER = ("srt:ConsolidationItemsAxis", "us-gaap:OperatingSegmentsMember")
# Members add up to the total when within this share of it.
_ADDS_UP_TOLERANCE = 0.005


class Figure(BaseModel):
    """One line item across the fiscal years, with where it came from."""

    model_config = ConfigDict(frozen=True)

    name: str
    unit: str
    concept: str | None
    label: str | None
    values: dict[int, float | None]


class Breakdown(BaseModel):
    """One concept split along one axis, e.g. revenue by product."""

    model_config = ConfigDict(frozen=True)

    of: str
    axis: str
    members: list[Figure]
    # True when, every year, the members sum to the total. False means the
    # members overlap or nest (AAPL "Products" and iPhone) or leave part out:
    # report them one by one and never add them up.
    adds_up_to_total: bool
    dropped: dict[str, str]


class FinancialHistory(BaseModel):
    """Headline figures for the fiscal years in one 10-K, newest first."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    accession: str
    fiscal_years: list[int]
    fiscal_year_ends: dict[int, dt.date]
    figures: dict[str, Figure]
    debt_components: list[Figure]
    revenue_breakdowns: list[Breakdown]
    unavailable: dict[str, str]
    source: str = "10-K XBRL"


def _facts(xbrl: Any, concept: str) -> pd.DataFrame:
    """Facts for one us-gaap concept with no dimensions.

    ``FactQuery.exclude_dimensions()`` is a no-op in edgartools 5.60.0 (AAPL
    revenue: 45 facts with and without it, iPhone and region breakdowns
    included), so totals are selected by the facts' ``is_dimensioned`` flag.
    """
    df = xbrl.query().by_concept(f"us-gaap:{concept}", exact=True).to_dataframe()
    if df.empty or "is_dimensioned" not in df:
        return pd.DataFrame()
    return df[~df["is_dimensioned"].astype(bool)]


def _annual(df: pd.DataFrame) -> pd.DataFrame:
    """Rows covering a full fiscal year."""
    if "period_start" not in df or "period_end" not in df:
        return df.iloc[0:0]
    rows = df.dropna(subset=["period_start", "period_end"]).copy()
    days = (
        pd.to_datetime(rows["period_end"]) - pd.to_datetime(rows["period_start"])
    ).dt.days
    return rows[(days >= _YEAR_DAYS[0]) & (days <= _YEAR_DAYS[1])]


def _one_value(rows: pd.DataFrame) -> float | None:
    """The value of one period's facts; duplicates must agree."""
    values = rows["numeric_value"].dropna().unique()
    if len(values) == 0:
        return None
    if len(values) > 1:
        raise ValueError(f"conflicting values for one period: {sorted(values)}")
    return float(values[0])


def fiscal_year_ends(xbrl: Any) -> dict[int, dt.date]:
    """The latest ``YEARS`` fiscal-year end dates, by fiscal year, newest first."""
    for concept in REVENUE:
        annual = _annual(_facts(xbrl, concept))
        if not annual.empty:
            ends = sorted(
                {dt.date.fromisoformat(str(e)[:10]) for e in annual["period_end"]},
                reverse=True,
            )[:YEARS]
            return {end.year: end for end in ends}
    raise LookupError("no annual revenue facts: can't tell the fiscal years")


def _duration_figure(
    xbrl: Any, name: str, unit: str, concepts: tuple[str, ...], ends: dict[int, dt.date]
) -> Figure | None:
    for concept in concepts:
        annual = _annual(_facts(xbrl, concept))
        if annual.empty:
            continue
        values = {
            year: _one_value(
                annual[annual["period_end"].astype(str).str[:10] == str(end)]
            )
            for year, end in ends.items()
        }
        if any(v is not None for v in values.values()):
            return Figure(
                name=name,
                unit=unit,
                concept=concept,
                label=str(annual["label"].iloc[0]),
                values=values,
            )
    return None


def _instant_figure(
    xbrl: Any, name: str, concepts: tuple[str, ...], ends: dict[int, dt.date]
) -> Figure | None:
    for concept in concepts:
        df = _facts(xbrl, concept)
        if df.empty or "period_instant" not in df:
            continue
        instants = df.dropna(subset=["period_instant"])
        values = {
            year: _one_value(
                instants[instants["period_instant"].astype(str).str[:10] == str(end)]
            )
            for year, end in ends.items()
        }
        if any(v is not None for v in values.values()):
            return Figure(
                name=name,
                unit="USD",
                concept=concept,
                label=str(instants["label"].iloc[0]),
                values=values,
            )
    return None


def _debt_components(xbrl: Any, ends: dict[int, dt.date]) -> list[Figure]:
    """The balance sheet's debt lines, as presented (2 years)."""
    sheet = xbrl.statements.balance_sheet().to_dataframe()
    lines = sheet[
        ~sheet["dimension"].astype(bool) & sheet["standard_concept"].isin(DEBT_LINES)
    ]
    components = []
    for _, row in lines.iterrows():
        values = {
            year: (
                float(row[str(end)])
                if str(end) in row and pd.notna(row[str(end)])
                else None
            )
            for year, end in ends.items()
        }
        components.append(
            Figure(
                name=row["standard_concept"],
                unit="USD",
                concept=row["concept"].split("_", 1)[1],
                label=row["label"],
                values=values,
            )
        )
    return components


def _member_label(xbrl: Any, member: str) -> str:
    """The company's label for a member, without "[Member]"; else from its name."""
    element = xbrl.element_catalog.get(member.replace(":", "_"))
    labels = getattr(element, "labels", None) or {}
    label = labels.get("http://www.xbrl.org/2003/role/label")
    if label:
        return label.removesuffix(" [Member]").strip()
    prefix, _, local = member.partition(":")
    name = local.removesuffix("Member")
    if prefix == "country":  # ISO codes from the country taxonomy: US, CN, TW
        return name
    return re.sub(
        r"(?<=[a-z])(?=[A-Z])", " ", name
    )  # OtherCountries -> Other Countries


def _breakdowns(
    xbrl: Any, revenue: Figure, ends: dict[int, dt.date]
) -> list[Breakdown]:
    facts = xbrl.facts.get_facts_with_dimensions()
    rows = _annual(facts[facts["concept"] == f"us-gaap:{revenue.concept}"])
    dim_columns = [c for c in rows.columns if c.startswith("dim_")]
    by_end = {str(end): year for year, end in ends.items()}

    # (axis, member) -> {year: set of values}
    found: dict[tuple[str, str], dict[int, set[float]]] = {}
    for _, row in rows.iterrows():
        year = by_end.get(str(row["period_end"])[:10])
        if year is None or pd.isna(row["numeric_value"]):
            continue
        dims = {c[4:]: row[c] for c in dim_columns if pd.notna(row[c])}
        dims = {_dim_name(axis): member for axis, member in dims.items()}
        if dims.get(_SEGMENT_QUALIFIER[0]) == _SEGMENT_QUALIFIER[1]:
            del dims[_SEGMENT_QUALIFIER[0]]
        if len(dims) != 1:
            continue
        ((axis, member),) = dims.items()
        if axis not in BREAKDOWN_AXES:
            continue
        found.setdefault((axis, member), {}).setdefault(year, set()).add(
            float(row["numeric_value"])
        )

    breakdowns = []
    for axis, name in BREAKDOWN_AXES.items():
        members, dropped = [], {}
        for (member_axis, member), by_year in found.items():
            if member_axis != axis:
                continue
            label = _member_label(xbrl, member)
            if any(len(v) > 1 for v in by_year.values()):
                dropped[label] = "conflicting values for one year"
                continue
            values = {
                y: (next(iter(by_year[y])) if y in by_year else None) for y in ends
            }
            members.append(
                Figure(
                    name=member,
                    unit="USD",
                    concept=revenue.concept,
                    label=label,
                    values=values,
                )
            )
        if not members and not dropped:
            continue
        adds_up = bool(members) and all(
            revenue.values.get(y)
            and all(m.values[y] is not None for m in members)
            and abs(sum(m.values[y] for m in members) - revenue.values[y])
            <= _ADDS_UP_TOLERANCE * abs(revenue.values[y])
            for y in ends
        )
        breakdowns.append(
            Breakdown(
                of="revenue",
                axis=name,
                members=members,
                adds_up_to_total=adds_up,
                dropped=dropped,
            )
        )
    return breakdowns


def _dim_name(column_suffix: str) -> str:
    """``srt_ProductOrServiceAxis`` -> ``srt:ProductOrServiceAxis``."""
    prefix, _, local = column_suffix.partition("_")
    return f"{prefix}:{local}"


def _ratio(
    numerator: Figure | None, denominator: Figure | None, years: list[int]
) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for year in years:
        n = numerator.values.get(year) if numerator else None
        d = denominator.values.get(year) if denominator else None
        out[year] = n / d if n is not None and d else None
    return out


def financial_history(filing: Any, ticker: str) -> FinancialHistory:
    """Headline figures for the fiscal years in ``filing``."""
    xbrl = filing.xbrl()
    meta = filing_meta(filing, ticker)
    ends = fiscal_year_ends(xbrl)
    years = list(ends)

    figures: dict[str, Figure] = {}
    unavailable: dict[str, str] = {}

    def add(key: str, figure: Figure | None, missing: str) -> None:
        if figure is None:
            unavailable[key] = missing
        else:
            figures[key] = figure

    add(
        "revenue",
        _duration_figure(xbrl, "revenue", "USD", REVENUE, ends),
        "no total revenue fact",
    )
    add(
        "gross_profit",
        _duration_figure(xbrl, "gross profit", "USD", GROSS_PROFIT, ends),
        "the company reports no gross profit",
    )
    add(
        "operating_income",
        _duration_figure(xbrl, "operating income", "USD", OPERATING_INCOME, ends),
        "the company reports no operating income",
    )
    add(
        "net_income",
        _duration_figure(xbrl, "net income", "USD", NET_INCOME, ends),
        "no net income fact",
    )
    add(
        "eps_diluted",
        _duration_figure(xbrl, "diluted EPS", "USD/share", EPS_DILUTED, ends),
        "no diluted EPS fact",
    )
    add(
        "cash",
        _instant_figure(xbrl, "cash, equivalents and restricted cash", CASH, ends),
        "no cash balance fact",
    )

    revenue = figures.get("revenue")
    for key, source, label in (
        ("gross_margin", "gross_profit", "gross margin"),
        ("operating_margin", "operating_income", "operating margin"),
    ):
        if source in figures and revenue:
            figures[key] = Figure(
                name=label,
                unit="ratio",
                concept=None,
                label=f"{figures[source].label} / {revenue.label} (computed)",
                values=_ratio(figures[source], revenue, years),
            )
        else:
            unavailable[key] = unavailable.get(source, "no revenue")

    components = _debt_components(xbrl, ends)
    if components:
        total = {
            year: (
                sum(c.values[year] for c in components)
                if all(c.values[year] is not None for c in components)
                else None
            )
            for year in years
        }
        figures["total_debt"] = Figure(
            name="total debt",
            unit="USD",
            concept=None,
            label="sum of the balance sheet's debt lines (computed)",
            values=total,
        )
        missing_years = [y for y, v in total.items() if v is None]
        if missing_years:
            unavailable["total_debt_years"] = (
                f"{missing_years}: the 10-K has two balance sheets; earlier debt is in the prior 10-K"
            )
    else:
        unavailable["total_debt"] = "no debt lines on the balance sheet"

    breakdowns = _breakdowns(xbrl, revenue, ends) if revenue else []

    return FinancialHistory(
        ticker=meta.ticker,
        accession=meta.accession,
        fiscal_years=years,
        fiscal_year_ends=ends,
        figures=figures,
        debt_components=components,
        revenue_breakdowns=breakdowns,
        unavailable=unavailable,
    )


def get_financial_history(ticker: str) -> FinancialHistory:
    """Headline figures for the three fiscal years in ``ticker``'s latest 10-K."""
    return financial_history(get_latest_10k(ticker), ticker)
