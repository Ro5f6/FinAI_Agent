"""Tests for the system prompt: it stays consistent with the tools, the
citation patterns the verifier will use, and the caching requirement."""

import re

from finai.agent.prompts import (
    FILING_CITATION,
    PORTFOLIO_CITATION,
    PRICE_CITATION,
    QUOTE_CITATION,
    SYSTEM_PROMPT,
)
from finai.ingestion.parse import TARGET_ITEMS


def test_the_example_citations_match_the_verifier_patterns() -> None:
    """The prompt's examples and the patterns the verifier parses can't drift."""
    filing = FILING_CITATION.search(SYSTEM_PROMPT)
    quote = QUOTE_CITATION.search(SYSTEM_PROMPT)

    assert filing is not None
    assert filing.group("ticker", "fiscal_year", "section") == (
        "AAPL",
        "2025",
        "Item 1A",
    )
    assert quote is not None
    assert quote.group("quote_time") == "2026-10-09 20:00"
    assert PORTFOLIO_CITATION in SYSTEM_PROMPT
    price = PRICE_CITATION.search(SYSTEM_PROMPT)
    assert price is not None
    assert price.group("end_date") == "2026-10-14"
    assert "[AAPL 10-K FY2025, Item 8]" in SYSTEM_PROMPT


def test_price_citation_rejects_loose_forms() -> None:
    assert PRICE_CITATION.fullmatch("[Alpaca prices, to 2026-06-15]")
    for loose in (
        "[Alpaca, 2026-06-15]",
        "[Alpaca prices]",
        "(Alpaca prices, to 2026-06-15)",
    ):
        assert not PRICE_CITATION.fullmatch(loose), loose


def test_filing_citation_accepts_every_ingested_section() -> None:
    for section in TARGET_ITEMS:
        citation = f"[MSFT 10-K FY2026, {section}]"
        match = FILING_CITATION.fullmatch(citation)
        assert match is not None, section
        assert match.group("section") == section


def test_filing_citation_accepts_share_class_tickers() -> None:
    assert FILING_CITATION.fullmatch("[BRK.B 10-K FY2025, Item 7]")


def test_filing_citation_rejects_loose_forms() -> None:
    for loose in (
        "[AAPL 10-K 2025, Item 1A]",
        "[AAPL 10-K FY2025, Risk Factors]",
        "[AAPL FY2025 Item 1A]",
        "(AAPL 10-K FY2025, Item 1A)",
    ):
        assert not FILING_CITATION.fullmatch(loose), loose


def test_no_shouted_rules() -> None:
    """Current models over-apply capitalised MUST/CRITICAL instructions."""
    assert not re.search(r"\b(MUST|NEVER|ALWAYS|CRITICAL|IMPORTANT)\b", SYSTEM_PROMPT)


def test_prompt_is_static_for_caching() -> None:
    """A date or per-user value would change the prefix and defeat caching."""
    assert "{" not in SYSTEM_PROMPT  # no unfilled template fields
    assert "today" not in SYSTEM_PROMPT.lower()


def test_prompt_never_discourages_tool_use() -> None:
    """Sonnet 5.5 follows 'minimise tool calls' literally."""
    lowered = SYSTEM_PROMPT.lower()
    for phrase in ("minimi", "only use tools", "avoid calling", "sparingly"):
        assert phrase not in lowered
