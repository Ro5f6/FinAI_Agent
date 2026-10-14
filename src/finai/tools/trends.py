"""Trends computed in Python from a 10-K's figures (1.3.23).

The model never does arithmetic (CLAUDE.md), so growth and margin changes are
computed here, from ``FinancialHistory`` (1.3.22), and the agent quotes them.

Definitions, chosen so a quoted figure can't be misread:

* **Change** is the difference in the figure's own unit (dollars, dollars per
  share). Always given when both years have a value.
* **Growth** is ``current / earlier - 1``, only when both values are positive.
  A loss in either year makes it not meaningful: Ford's net income going from
  +$5.88B to -$8.18B is -239% arithmetically; Rivian's loss narrowing from
  -$4.75B to -$3.65B would read as -23% "growth" although it improved.
* **The span** covers the filing's oldest to newest year: two annual steps
  from three fiscal years. It carries the total growth and the compound
  annual growth rate (CAGR), with both years named, so "over two years" and
  "per year" can't be confused.
* **Margin changes** are in percentage points: 46.2% to 46.9% is +0.7 pp,
  not +1.5%.
"""

from itertools import pairwise

from pydantic import BaseModel, ConfigDict

from finai.tools.financials import Figure, FinancialHistory

GROWTH_FIGURES = ("revenue", "net_income", "eps_diluted")
MARGIN_FIGURES = ("gross_margin", "operating_margin")

NOT_MEANINGFUL = "not meaningful: a loss or zero in one of the years"
MISSING = "a year's value is unavailable"


class Change(BaseModel):
    """How one figure moved between two fiscal years."""

    model_config = ConfigDict(frozen=True)

    from_year: int
    to_year: int
    years: int
    from_value: float | None
    to_value: float | None
    unit: str
    # In the figure's unit; for margins, in percentage points.
    change: float | None
    # Fractions (0.064 = 6.4%); None with ``note`` when not meaningful.
    growth: float | None = None
    cagr: float | None = None
    note: str | None = None


class Trends(BaseModel):
    """Year-over-year and span changes for a company's headline figures."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    fiscal_years: list[int]
    # figure -> changes, newest first: each year over the previous, then the span.
    year_over_year: dict[str, list[Change]]
    span: dict[str, Change]
    unavailable: dict[str, str]


def _change(figure: Figure, earlier: int, later: int) -> Change:
    a, b = figure.values.get(earlier), figure.values.get(later)
    years = later - earlier
    common = {
        "from_year": earlier,
        "to_year": later,
        "years": years,
        "from_value": a,
        "to_value": b,
    }
    if a is None or b is None:
        return Change(**common, unit=figure.unit, change=None, note=MISSING)

    if figure.unit == "ratio":  # margins: percentage points, no growth rate
        return Change(**common, unit="percentage points", change=(b - a) * 100)

    change = b - a
    if a <= 0 or b <= 0:
        return Change(**common, unit=figure.unit, change=change, note=NOT_MEANINGFUL)
    growth = b / a - 1
    cagr = (b / a) ** (1 / years) - 1 if years > 1 else None
    return Change(**common, unit=figure.unit, change=change, growth=growth, cagr=cagr)


def compute_trends(history: FinancialHistory) -> Trends:
    """Trends for the fiscal years in one ``FinancialHistory``."""
    years = sorted(history.fiscal_years)  # oldest first for pairing
    year_over_year: dict[str, list[Change]] = {}
    span: dict[str, Change] = {}
    unavailable: dict[str, str] = {}

    for key in (*GROWTH_FIGURES, *MARGIN_FIGURES):
        figure = history.figures.get(key)
        if figure is None:
            unavailable[key] = history.unavailable.get(key, "not in the filing")
            continue
        pairs = list(pairwise(years))
        year_over_year[key] = [_change(figure, a, b) for a, b in reversed(pairs)]
        if len(years) > 2:
            span[key] = _change(figure, years[0], years[-1])

    return Trends(
        ticker=history.ticker,
        fiscal_years=history.fiscal_years,
        year_over_year=year_over_year,
        span=span,
        unavailable=unavailable,
    )
