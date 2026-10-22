"""Deterministic checks of a draft answer against what the tools returned (1.5).

They run before the LLM critic, cost nothing and give the same result every
time. A failure sends the draft back to the planner with the reasons.

Numbers (1.5.4)
---------------
Every figure in the draft must be a value some tool returned, rounded no
further than the draft shows it. A figure is a number written with ``$`` or
``%``, a scale word (thousand, million, billion, trillion), a decimal point,
thousands separators, or "x"/"times"; bare small integers ("Item 7", "two
risks") aren't figures. Numbers joined to letters or symbols are names or
dates, not figures: H200, A100, 10-K, FY2026, Q1, 52-week, D:5, 2026-07-26,
18:01. Citations in square brackets are skipped.

The sources are every number in the thread's tool results, JSON fields and
numbers inside text (10-K passages) alike, and in the user's question. A
number in text without a scale word may be a table value "in millions", so
it is also tried in thousands, millions and billions.

A figure matches a source when the source rounds to it at the precision
shown: "26.9" matches 26.8975, "about $5.25 trillion" matches
5,247,534,028,858. A percentage also matches a fraction ("6.4%" and
0.0642551: tools return rates as fractions, D-036). Signs are ignored:
"down 4.14%" carries the sign in words and the quote has -4.14.

Gap, as the plan notes: a figure that appears in a retrieved passage passes
even if the answer gives it the wrong year. The year-labelled XBRL values
from ``get_financial_history`` would allow checking the period too.
Citations (1.5.5)
-----------------
Every citation must name a source a tool returned in this thread, in one of
the forms the system prompt asks for (``prompts``):

* ``[NVDA 10-K FY2026, Item 1A]``: a retrieved passage with that ticker,
  fiscal year and section, or a ``get_financial_history`` result whose
  ``citation`` is exactly this (its Item 8 figures);
* ``[Finnhub quote, 2026-10-22 20:00 UTC]``: a quote with that time, to
  the minute;
* ``[Alpaca prices, to 2026-10-14]``: a price history ending that day;
* ``[portfolio]``: the portfolio tool was called.

A bracket that mentions a source ("10-K", "Finnhub", "Alpaca", "portfolio")
but matches none of these forms is flagged too: the verifier can't check
it. Markdown links, ``[text](url)``, aren't citations.
"""

import math
import re
from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict

from finai.agent.prompts import (
    FILING_CITATION,
    PORTFOLIO_CITATION,
    PRICE_CITATION,
    QUOTE_CITATION,
)

SCALES: dict[str, float] = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}
# Table values in 10-K text are often in thousands or millions.
_UNSTATED_SCALES = (1.0, 1e3, 1e6, 1e9)

_NUMBER = re.compile(
    r"""
    (?<![\w.:/\-])                       # not part of a name, date, time or code
    (?P<cur>\$\s?)?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?!(?!x\b)\w|[-/:]\d|-[A-Za-z])     # nor followed by one (10-K, 2026-07, 18:01); 26.9x is fine
    (?P<pct>\s?%)?
    (?:\s?(?P<scale>thousand|million|billion|trillion)s?\b)?
    (?:\s?(?P<times>x|times)\b)?
    """,
    re.VERBOSE | re.IGNORECASE,
)
_CITATION = re.compile(r"\[[^\]]*\]")


class Issue(BaseModel):
    """One problem with a draft, as the planner is told about it."""

    model_config = ConfigDict(frozen=True)

    claim: str
    reason: str


class Figure(NamedTuple):
    text: str  # as written in the draft
    value: float  # absolute, scaled
    tolerance: float  # half a unit of the last digit shown, scaled
    percent: bool


def _figure(match: re.Match[str]) -> Figure:
    digits = match["num"].replace(",", "")
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    scale = SCALES[match["scale"].lower()] if match["scale"] else 1.0
    return Figure(
        text=match.group(0).strip(),
        value=abs(float(digits)) * scale,
        tolerance=0.5 * 10**-decimals * scale,
        percent=bool(match["pct"]),
    )


def figures(text: str) -> list[Figure]:
    """The figures in ``text`` that the check covers, citations removed."""
    found = []
    for match in _NUMBER.finditer(_CITATION.sub(" ", text)):
        checkable = (
            match["cur"]
            or match["pct"]
            or match["scale"]
            or match["times"]
            or "." in match["num"]
            or "," in match["num"]
        )
        if checkable:
            found.append(_figure(match))
    return found


def _numbers_in_text(text: str) -> Iterator[float]:
    for match in _NUMBER.finditer(text):
        figure = _figure(match)
        if match["scale"]:
            yield figure.value
        else:
            yield from (figure.value * s for s in _UNSTATED_SCALES)


def source_values(sources: Iterable[Any]) -> list[float]:
    """Every number in the given tool outputs and texts, absolute."""
    values: list[float] = []

    def walk(item: Any) -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, int | float):
            if math.isfinite(item):
                values.append(abs(float(item)))
        elif isinstance(item, str):
            values.extend(_numbers_in_text(item))
        elif isinstance(item, dict):
            for value in item.values():
                walk(value)
        elif isinstance(item, list | tuple):
            for value in item:
                walk(value)

    for source in sources:
        walk(source)
    return values


def _matches(figure: Figure, value: float) -> bool:
    candidates = [figure.value]
    tolerances = [figure.tolerance]
    if figure.percent:  # tools give rates as fractions (D-036)
        candidates.append(figure.value / 100)
        tolerances.append(figure.tolerance / 100)
    return any(
        abs(value - c) <= t + 1e-9 * max(c, 1.0)
        for c, t in zip(candidates, tolerances, strict=True)
    )


def _nearest(figure: Figure, values: list[float]) -> float | None:
    target = figure.value / 100 if figure.percent and figure.value > 1 else figure.value
    return min(values, key=lambda v: abs(v - target), default=None)


def check_numbers(
    draft: str, tool_results: Iterable[dict[str, Any]], question: str = ""
) -> list[Issue]:
    """An issue for each figure in ``draft`` that no tool result supports."""
    values = source_values([r.get("output") for r in tool_results])
    values += source_values([question])
    issues = []
    for figure in figures(draft):
        if any(_matches(figure, v) for v in values):
            continue
        nearest = _nearest(figure, values)
        hint = f" (nearest tool value: {nearest:,.6g})" if nearest is not None else ""
        issues.append(
            Issue(
                claim=figure.text,
                reason=f"{figure.text} is not in any tool result{hint}",
            )
        )
    return issues


_BRACKET = re.compile(r"\[[^\[\]]*\](?!\()")
_SOURCE_WORDS = re.compile(r"10-K|Finnhub|Alpaca|portfolio", re.IGNORECASE)


def _successful(
    tool_results: Iterable[dict[str, Any]], tool: str
) -> list[dict[str, Any]]:
    return [
        r["output"]
        for r in tool_results
        if r.get("tool") == tool
        and r.get("status") == "success"
        and isinstance(r.get("output"), dict)
    ]


def check_citations(
    draft: str,
    tool_results: Iterable[dict[str, Any]],
    retrieved_docs: Iterable[dict[str, Any]],
) -> list[Issue]:
    """An issue for each citation that names a source no tool returned."""
    results = list(tool_results)
    passages = {
        (
            str(d.get("ticker", "")).upper(),
            int(d.get("fiscal_year", 0)),
            d.get("section"),
        )
        for d in retrieved_docs
    }
    filing_citations = {
        o.get("citation") for o in _successful(results, "get_financial_history")
    }
    quote_times = {
        str(o.get("quote_time", ""))[:16].replace("T", " ")
        for o in _successful(results, "get_live_market_data")
    }
    price_ends = {o.get("end_date") for o in _successful(results, "get_price_history")}
    has_portfolio = bool(_successful(results, "get_user_portfolio"))

    issues = []
    seen = set()
    for match in _BRACKET.finditer(draft):
        citation = match.group(0)
        if citation in seen:
            continue
        seen.add(citation)
        if filing := FILING_CITATION.fullmatch(citation):
            key = (filing["ticker"], int(filing["fiscal_year"]), filing["section"])
            if key not in passages and citation not in filing_citations:
                issues.append(
                    Issue(
                        claim=citation,
                        reason=f"{citation}: no {key[0]} FY{key[1]} {key[2]} passage "
                        "or financial history was retrieved in this conversation",
                    )
                )
        elif quote := QUOTE_CITATION.fullmatch(citation):
            if quote["quote_time"] not in quote_times:
                issues.append(
                    Issue(
                        claim=citation,
                        reason=f"{citation}: no quote with that time was returned",
                    )
                )
        elif price := PRICE_CITATION.fullmatch(citation):
            if price["end_date"] not in price_ends:
                issues.append(
                    Issue(
                        claim=citation,
                        reason=f"{citation}: no price history ending that day was returned",
                    )
                )
        elif citation == PORTFOLIO_CITATION:
            if not has_portfolio:
                issues.append(
                    Issue(
                        claim=citation,
                        reason="[portfolio] cited but the portfolio wasn't read",
                    )
                )
        elif _SOURCE_WORDS.search(citation):
            issues.append(
                Issue(
                    claim=citation,
                    reason=f"{citation} isn't in a citation form the verifier can check",
                )
            )
    return issues
