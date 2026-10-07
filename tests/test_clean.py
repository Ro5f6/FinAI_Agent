"""Tests for cleaning extracted 10-K items.

Fixtures reproduce the layouts measured on real filings (see the module
docstring of ``finai.ingestion.clean``) rather than invented ones.
"""

import re

import pytest

from finai.ingestion.clean import (
    MIN_SECTION_CHARS,
    clean_items,
    clean_section,
    find_running_headers,
)

_SENTENCE = "The Company depends on component suppliers located outside the US."


def _pages(bodies: list[str], headers: list[list[str]], first_page: int = 1) -> str:
    """Join page bodies with a blank-isolated page number and the next page's headers.

    ``headers[i]`` are the running header lines printed above ``bodies[i + 1]``.
    """
    parts = [bodies[0]]
    for n, (body, top) in enumerate(zip(bodies[1:], headers, strict=True)):
        parts.append(str(first_page + n))
        parts.extend(top)
        parts.append(body)
    return "\n\n".join(parts)


# --- whitespace ------------------------------------------------------------


def test_unicode_spaces_become_single_spaces() -> None:
    raw = "Item 1A.\xa0\xa0\xa0\xa0Risk Factors\n\nNet\xa0sales\u2009rose \u202f 5%.\u200b"

    assert clean_section(raw) == "Net sales rose 5%."


def test_paragraph_breaks_and_table_rows_are_preserved() -> None:
    raw = (
        "Item 7. MD&A\n\nFirst paragraph.\n\n\n\nSecond paragraph.\n\n"
        "Year  2025  2024\nNet sales  $  416,161  $  391,035"
    )

    assert clean_section(raw) == (
        "First paragraph.\n\nSecond paragraph.\n\n"
        "Year 2025 2024\nNet sales $ 416,161 $ 391,035"
    )


def test_footnote_rule_lines_are_dropped() -> None:
    raw = "Item 1. Business\n\nTotal 4,413\n\n__________\n\n(a) Includes trucks."

    assert "_" not in clean_section(raw)


# --- the opening item heading ----------------------------------------------


@pytest.mark.parametrize(
    "opening",
    [
        "Item 1A.\xa0\xa0\xa0\xa0Risk Factors",  # AAPL
        "ITEM 1A. RISK FACTORS",  # MSFT
        "ITEM 1A. Risk Factors. ",  # F
        "RIVIAN AUTOMOTIVE, INC.\n\nItem 1A. Risk Factors",  # header above it
        "Parts II and III\n\nItem 1A. Risk Factors.",  # JPM
    ],
)
def test_opening_item_heading_is_removed(opening: str) -> None:
    assert clean_section(f"{opening}\n\n{_SENTENCE}") == _SENTENCE


def test_item_reference_deep_in_the_text_is_kept() -> None:
    """Only the opening lines are searched; later "Item 7" lines are content."""
    raw = "Intro.\n\nMore.\n\nFurther.\n\nItem 7 discusses margins."

    assert clean_section(raw).endswith("Item 7 discusses margins.")


# --- page footers ----------------------------------------------------------


def test_bare_page_numbers_are_removed() -> None:
    raw = _pages(["Para one.", "Para two.", "Para three."], [[], []], first_page=14)

    assert clean_section(raw) == "Para one.\n\nPara two.\n\nPara three."


def test_apple_style_footer_is_removed() -> None:
    raw = "First para.\n\nApple Inc. | 2025 Form 10-K | 5\n\nSecond para."

    assert clean_section(raw) == "First para.\n\nSecond para."


def test_number_inside_a_table_is_kept() -> None:
    """Table rows are separated by single newlines, so a lone cell is not a page."""
    raw = "Stores by region\nAmericas\n412\nEurope\n98"

    assert clean_section(raw) == raw


# --- running headers -------------------------------------------------------


def test_continued_headers_are_removed() -> None:
    """F repeats the item title with "(Continued)" atop every page."""
    top = ["Item 1A. Risk Factors (Continued)"]
    raw = _pages([f"Risk {i}." for i in range(5)], [top] * 4)

    cleaned = clean_section(raw, find_running_headers([raw]))

    assert "Continued" not in cleaned
    assert cleaned == "\n\n".join(f"Risk {i}." for i in range(5))


def test_stacked_two_line_headers_are_removed() -> None:
    """MSFT prints "PART II" then "Item 8" atop each page."""
    raw = _pages([f"Note {i}." for i in range(5)], [["PART II", "Item 8"]] * 4)

    cleaned = clean_section(raw, find_running_headers([raw]))

    assert cleaned == "\n\n".join(f"Note {i}." for i in range(5))


def test_subsection_that_sometimes_starts_a_page_is_kept() -> None:
    """F's "Ford Credit Segment" starts 3 of ~50 pages; it is a real heading."""
    title = "Item 7. MD&A (Continued)"
    tops = [
        [title, "Ford Credit Segment"] if i % 10 == 0 else [title] for i in range(30)
    ]
    raw = _pages([f"Body {i}." for i in range(31)], tops)

    headers = find_running_headers([raw])

    assert title in headers
    assert "Ford Credit Segment" not in headers
    assert clean_section(raw, headers).count("Ford Credit Segment") == 3


def test_table_heading_that_also_appears_mid_page_is_kept() -> None:
    """AAPL's "2025 2024 2023" tops 3 pages but occurs 15 times overall."""
    columns = "2025  2024  2023"
    bodies = [f"Statement {i}\n{columns}\nRow {i}  1  2  3" for i in range(16)]
    tops = [["Apple Inc.", columns] if i < 3 else ["Apple Inc."] for i in range(15)]
    raw = _pages(bodies, tops)

    headers = find_running_headers([raw])

    assert "2025 2024 2023" not in headers
    assert "Apple Inc." in headers


def test_headers_are_learned_across_a_filings_sections() -> None:
    """JPM's "Part I" tops one page per short section but many in the filing."""
    sections = [_pages([f"A{i}.", f"B{i}."], [["Part I"]]) for i in range(3)]

    assert find_running_headers(sections[:1]) == frozenset()
    assert find_running_headers(sections) == {"Part I"}


# --- paragraphs split by a page break --------------------------------------


def test_sentence_cut_by_a_page_break_is_rejoined() -> None:
    top = ["Item 1A. Risk Factors (Continued)"]
    # Four pages, so the header repeats often enough to be learned.
    raw = _pages(
        ["Intro.", "Middle.", "a reduction or clawback of", "incentives); disrupt."],
        [top, top, top],
    )

    cleaned = clean_section(raw, find_running_headers([raw]))

    assert "a reduction or clawback of incentives); disrupt." in cleaned


def test_page_break_between_sentences_is_not_joined() -> None:
    raw = _pages(["First sentence ends here.", "second starts lower."], [[]])

    assert clean_section(raw) == "First sentence ends here.\n\nsecond starts lower."


def test_page_break_before_a_new_heading_is_not_joined() -> None:
    raw = _pages(["An unfinished table caption", "Liquidity and Capital"], [[]])

    assert "\n\nLiquidity and Capital" in clean_section(raw)


# --- vestigial sections ----------------------------------------------------


def test_cross_reference_sections_are_dropped_and_reported() -> None:
    nvda_item_8 = (
        "Item 8. Financial Statements and Supplementary Data\n\n"
        "The information required by this Item is set forth in our Consolidated "
        "Financial Statements and Notes thereto included in this Annual Report on "
        "Form 10-K."
    )
    items = {
        "Item 1A": f"Item 1A. Risk Factors\n\n{_SENTENCE * 40}",
        "Item 8": nvda_item_8,
    }

    result = clean_items(items)

    assert set(result.sections) == {"Item 1A"}
    assert set(result.vestigial) == {"Item 8"}
    assert result.vestigial["Item 8"] < MIN_SECTION_CHARS


def test_length_is_measured_after_cleaning() -> None:
    """Page furniture must not lift a cross-reference over the floor."""
    padding = "\n\n".join(f"{n}\n\nTable of Contents" for n in range(1, 400))
    items = {"Item 8": f"Item 8. Financial Statements\n\nSee page 108.\n\n{padding}"}

    assert clean_items(items).vestigial == {"Item 8": len("See page 108.")}


def test_floor_sits_between_measured_cross_references_and_real_sections() -> None:
    longest_cross_reference = 777  # XOM Item 8
    shortest_real_section = 1_988  # MSFT Item 7A

    assert longest_cross_reference < MIN_SECTION_CHARS < shortest_real_section


# --- live filings ----------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("ticker", ["AAPL", "NVDA", "MSFT", "F", "JPM"])
def test_clean_items_against_live_filings(ticker: str) -> None:
    """Five filers covering the footer and header styles measured."""
    from finai.ingestion.fetch import get_latest_10k
    from finai.ingestion.parse import extract_filing_items

    result = clean_items(extract_filing_items(get_latest_10k(ticker)))

    assert "Item 1A" in result.sections
    for item, text in result.sections.items():
        assert "\xa0" not in text
        assert not re.search(r"(^|\n\n)\d{1,3}(\n\n|$)", text), f"{ticker} {item}"
        assert not re.match(r"item\s*\d", text, re.IGNORECASE), f"{ticker} {item}"
