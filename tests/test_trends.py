"""Tests for trend computation (1.3.23), against hand-calculated values."""

import datetime as dt

import pytest

from finai.tools.financials import Figure, FinancialHistory
from finai.tools.trends import MISSING, NOT_MEANINGFUL, compute_trends


def _history(**figures: dict[int, float | None]) -> FinancialHistory:
    units = {
        "eps_diluted": "USD/share",
        "gross_margin": "ratio",
        "operating_margin": "ratio",
    }
    return FinancialHistory(
        ticker="TEST",
        accession="0000000000-25-000001",
        fiscal_years=[2025, 2024, 2023],
        fiscal_year_ends={
            2025: dt.date(2025, 12, 31),
            2024: dt.date(2024, 12, 31),
            2023: dt.date(2023, 12, 31),
        },
        figures={
            k: Figure(name=k, unit=units.get(k, "USD"), concept=None, label=k, values=v)
            for k, v in figures.items()
        },
        debt_components=[],
        revenue_breakdowns=[],
        unavailable={},
    )


# Apple FY2023-2025 revenue and net income, $ millions as printed in the 10-K.
APPLE_REVENUE = {2025: 416_161.0, 2024: 391_035.0, 2023: 383_285.0}
APPLE_NET_INCOME = {2025: 112_010.0, 2024: 93_736.0, 2023: 96_995.0}


def test_year_over_year_growth_matches_hand_calculation() -> None:
    t = compute_trends(_history(revenue=APPLE_REVENUE))
    latest, earlier = t.year_over_year["revenue"]

    # 416,161 / 391,035 - 1 = 0.0642551178; 391,035 / 383,285 - 1 = 0.0202199408
    assert (latest.from_year, latest.to_year) == (2024, 2025)
    assert latest.growth == pytest.approx(0.0642551178, rel=1e-8)
    assert latest.change == 25_126.0
    assert (earlier.from_year, earlier.to_year) == (2023, 2024)
    assert earlier.growth == pytest.approx(0.0202199408, rel=1e-8)


def test_span_gives_total_growth_and_cagr_with_its_years() -> None:
    span = compute_trends(_history(revenue=APPLE_REVENUE)).span["revenue"]

    # 416,161 / 383,285 - 1 = 0.0857742933; CAGR = sqrt(1.0857742933) - 1 = 0.0420049392
    assert (span.from_year, span.to_year, span.years) == (2023, 2025, 2)
    assert span.growth == pytest.approx(0.0857742933, rel=1e-8)
    assert span.cagr == pytest.approx(0.0420049392, rel=1e-8)
    assert span.change == 32_876.0


def test_one_year_change_has_no_cagr() -> None:
    latest = compute_trends(_history(revenue=APPLE_REVENUE)).year_over_year["revenue"][
        0
    ]

    assert latest.cagr is None


def test_falling_figure_gives_negative_growth() -> None:
    earlier = compute_trends(_history(net_income=APPLE_NET_INCOME)).year_over_year[
        "net_income"
    ][1]

    # 93,736 / 96,995 - 1 = -0.033599...
    assert earlier.growth == pytest.approx(-0.0335996, rel=1e-5)
    assert earlier.change == -3_259.0


@pytest.mark.parametrize(
    ("values", "change"),
    [
        (
            {2025: -8_182.0, 2024: 5_879.0, 2023: 4_347.0},
            -14_061.0,
        ),  # F: profit to loss
        (
            {2025: -3_646.0, 2024: -4_747.0, 2023: -5_432.0},
            1_101.0,
        ),  # RIVN: loss narrows
        ({2025: 10.0, 2024: 0.0, 2023: 5.0}, 10.0),  # zero base
    ],
)
def test_growth_is_not_meaningful_across_a_loss(values, change) -> None:
    """RIVN's narrowing loss would read as -23% "growth" although it improved."""
    latest = compute_trends(_history(net_income=values)).year_over_year["net_income"][0]

    assert latest.growth is None
    assert latest.note == NOT_MEANINGFUL
    assert latest.change == change


def test_margin_changes_are_percentage_points() -> None:
    """46.2% to 46.9% is +0.7 pp, not +1.5%."""
    t = compute_trends(_history(gross_margin={2025: 0.469, 2024: 0.462, 2023: 0.441}))
    latest = t.year_over_year["gross_margin"][0]

    assert latest.unit == "percentage points"
    assert latest.change == pytest.approx(0.7)
    assert latest.growth is None
    assert t.span["gross_margin"].change == pytest.approx(2.8)


def test_eps_keeps_its_unit() -> None:
    latest = compute_trends(
        _history(eps_diluted={2025: 7.46, 2024: 6.08, 2023: 6.13})
    ).year_over_year["eps_diluted"][0]

    assert latest.unit == "USD/share"
    assert latest.change == pytest.approx(1.38)
    assert latest.growth == pytest.approx(7.46 / 6.08 - 1)


def test_missing_year_is_reported_not_guessed() -> None:
    latest = compute_trends(
        _history(revenue={2025: 10.0, 2024: None, 2023: 8.0})
    ).year_over_year["revenue"][0]

    assert latest.change is None
    assert latest.note == MISSING


def test_figures_absent_from_the_filing_carry_their_reason() -> None:
    history = _history(revenue=APPLE_REVENUE).model_copy(
        update={"unavailable": {"gross_margin": "the company reports no gross profit"}}
    )

    t = compute_trends(history)

    assert t.unavailable["gross_margin"] == "the company reports no gross profit"
    assert "operating_margin" in t.unavailable


@pytest.mark.network
def test_live_apple_revenue_growth() -> None:
    from finai.tools.financials import get_financial_history

    t = compute_trends(get_financial_history("AAPL"))

    assert t.year_over_year["revenue"][0].growth == pytest.approx(
        416_161 / 391_035 - 1, rel=1e-6
    )
