"""Tests for table-aware chunking.

Fixtures copy the shapes of real 10-K tables (AAPL net sales, F statements);
small token limits make the splitting visible.
"""

import re
from itertools import pairwise

import pytest

from finai.ingestion.chunk import (
    ChunkPayload,
    chunk_filing,
    chunk_section,
    count_tokens,
)
from finai.ingestion.fetch import FilingMeta

_SENTENCE = "The Company depends on component suppliers located outside the US. "


def _prose(n_sentences: int, label: str = "") -> str:
    return " ".join(
        f"{label}Sentence {i} about supply risk." for i in range(n_sentences)
    )


_NET_SALES = (
    "The following table shows net sales by category (dollars in millions):\n\n"
    "2025  Change  2024  Change  2023\n"
    "iPhone  $  209,586  4%  $  201,183  —%  $  200,583\n"
    "Mac  33,708  12%  29,984  2%  29,357\n"
    "iPad  28,023  5%  26,694  (6)%  28,300"
)


def _long_table(n_groups: int = 4, rows_per_group: int = 12) -> str:
    """A statement with a unit line, year headings and labelled row groups."""
    lines = ["Years ended December 31,  2025  2024"]
    for g in range(n_groups):
        lines.append(f"Level {g}:")
        lines += [
            f"Line item {g}-{r}  {r},{100 + r}  {r},{200 + r}"
            for r in range(rows_per_group)
        ]
    return "CONSOLIDATED BALANCE SHEET\n\n(in millions)\n\n" + "\n".join(lines)


# --- size -------------------------------------------------------------------


def test_every_chunk_is_within_the_limit() -> None:
    text = "\n\n".join([_prose(30), _NET_SALES, _long_table(), _prose(80)])

    chunks = chunk_section(text, max_tokens=120, overlap_tokens=20)

    assert len(chunks) > 3
    assert all(count_tokens(c) <= 120 for c in chunks)


def test_short_paragraphs_are_packed_together() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    assert chunk_section(text, max_tokens=100, overlap_tokens=10) == [text]


def test_long_paragraph_is_split_at_sentence_ends() -> None:
    chunks = chunk_section(_prose(60), max_tokens=80, overlap_tokens=0)

    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)


# --- tables ---------------------------------------------------------------


def test_table_that_fits_stays_whole_with_its_caption() -> None:
    text = f"{_prose(40)}\n\n{_NET_SALES}\n\n{_prose(40)}"

    chunks = chunk_section(text, max_tokens=150, overlap_tokens=0)

    holding = [c for c in chunks if "209,586" in c]
    assert len(holding) == 1
    assert "(dollars in millions)" in holding[0]
    assert "2025  Change  2024" in holding[0]
    assert "iPad  28,023" in holding[0]


def test_oversized_table_pieces_repeat_caption_headings_and_group() -> None:
    chunks = chunk_section(_long_table(), max_tokens=120, overlap_tokens=20)

    assert len(chunks) > 2
    for piece in chunks:
        assert piece.startswith("CONSOLIDATED BALANCE SHEET\n\n(in millions)")
        assert "Years ended December 31,  2025  2024" in piece
    # A piece that starts inside a group says which group it continues.
    continued = [c for c in chunks if "(continued)" in c]
    assert continued
    for piece in continued:
        label = re.search(r"Level (\d+): \(continued\)", piece)
        assert label is not None
        first_row = re.search(r"Line item (\d+)-", piece)
        assert first_row is not None and first_row.group(1) == label.group(1)


def test_every_table_row_survives_splitting() -> None:
    table = _long_table()
    chunks = chunk_section(table, max_tokens=120, overlap_tokens=20)

    for row in table.split("\n"):
        if row:
            assert any(row in c for c in chunks), row


def test_tables_separated_only_by_a_blank_line_are_one_table() -> None:
    """F prints one statement as several blocks."""
    top = "Revenue  2025  2024\nAutomotive  $  1,000  $  900"
    bottom = "Ford Credit  200  180\nTotal  $  1,200  $  1,080"
    text = f"{_prose(40)}\n\n{top}\n\n{bottom}\n\n{_prose(40)}"

    chunks = chunk_section(text, max_tokens=150, overlap_tokens=0)

    holding = [c for c in chunks if "Automotive" in c]
    assert len(holding) == 1
    assert "Total  $  1,200" in holding[0]


def test_statement_footer_is_not_the_next_tables_caption() -> None:
    footer = "See accompanying Notes to Consolidated Financial Statements."
    second = "Cash flows  2025  2024\nOperating  $  5,000  $  4,000"
    text = f"Income  2025  2024\nNet income  $  1,000  $  900\n\n{footer}\n\n{second}"

    chunks = chunk_section(text, max_tokens=40, overlap_tokens=0)

    second_chunk = next(c for c in chunks if "Operating" in c)
    assert footer not in second_chunk


def test_label_row_is_not_a_heading() -> None:
    """``Net sales:`` names the rows below it. As a heading it would be repeated
    on every piece, labelling cost-of-sales rows as net sales."""
    rows = "\n".join(f"Region {i}  {i},{100 + i}  {i},{200 + i}" for i in range(30))
    table = f"2025  2024\nNet sales:\n{rows}\nCost of sales:\n{rows}"

    chunks = chunk_section(table, max_tokens=80, overlap_tokens=0)

    assert all(c.startswith("2025  2024\n") for c in chunks)
    cost_pieces = [c for c in chunks if "Cost of sales:" in c]
    assert len(cost_pieces) > 1
    assert all("Net sales:" not in c for c in cost_pieces[1:])


# --- overlap --------------------------------------------------------------


def test_consecutive_prose_chunks_share_whole_sentences() -> None:
    paragraphs = [_prose(6, label=f"P{p} ") for p in range(6)]
    chunks = chunk_section("\n\n".join(paragraphs), max_tokens=90, overlap_tokens=25)

    assert len(chunks) > 2
    for prev, nxt in pairwise(chunks):
        opening = nxt.split("\n\n")[0]
        assert opening.endswith(".")
        assert opening in prev  # carried over from the previous chunk
        assert count_tokens(opening) <= 25


def test_table_rows_are_not_repeated_as_overlap() -> None:
    text = f"{_prose(5)}\n\n{_NET_SALES}\n\n{_prose(30)}"

    chunks = chunk_section(text, max_tokens=110, overlap_tokens=25)

    assert sum("209,586" in c for c in chunks) == 1


def test_overlap_alone_is_never_a_chunk() -> None:
    chunks = chunk_section("\n\n".join(_prose(6, f"P{p} ") for p in range(4)), 90, 25)

    last = chunks[-1].split("\n\n")
    assert not (len(last) == 1 and last[0] in chunks[-2])


def test_empty_section_gives_no_chunks() -> None:
    assert chunk_section("", max_tokens=100, overlap_tokens=10) == []


def test_defaults_come_from_settings() -> None:
    from finai.config import settings

    text = "\n\n".join(_prose(8, f"P{p} ") for p in range(40))
    chunks = chunk_section(text)

    assert all(count_tokens(c) <= settings.CHUNK_TOKENS for c in chunks)
    assert max(count_tokens(c) for c in chunks) > settings.CHUNK_TOKENS * 0.8


# --- payload ----------------------------------------------------------------

_META = FilingMeta(
    ticker="AAPL",
    accession="0000320193-25-000079",
    form="10-K",
    filing_date="2025-10-31",
    period_of_report="2025-09-27",
    fiscal_year=2025,
)


def test_payload_carries_the_planned_fields() -> None:
    [payload] = chunk_filing(_META, {"Item 1A": "Supply risk."}, 100, 10)

    assert payload.model_dump() == {
        "ticker": "AAPL",
        "form": "10-K",
        "fiscal_year": 2025,
        "filing_date": "2025-10-31",
        "section": "Item 1A",
        "chunk_index": 0,
        "accession": "0000320193-25-000079",
        "text": "Supply risk.",
    }


def test_chunk_index_restarts_in_each_section() -> None:
    sections = {"Item 1A": _prose(60, "A "), "Item 7": _prose(60, "B ")}

    payloads = chunk_filing(_META, sections, max_tokens=80, overlap_tokens=10)

    for section, body in sections.items():
        mine = [p for p in payloads if p.section == section]
        assert [p.chunk_index for p in mine] == list(range(len(mine)))
        assert [p.text for p in mine] == chunk_section(body, 80, 10)


def test_payload_is_immutable() -> None:
    [payload] = chunk_filing(_META, {"Item 1": "Business."}, 100, 10)

    with pytest.raises(ValueError):
        payload.fiscal_year = 2024  # type: ignore[misc]


def test_payload_rejects_a_non_integer_year() -> None:
    """Year filters use Qdrant's integer index (1.2.14)."""
    with pytest.raises(ValueError):
        ChunkPayload(**{**_payload_fields(), "fiscal_year": "FY2025"})


def _payload_fields() -> dict[str, object]:
    return chunk_filing(_META, {"Item 1": "Business."}, 100, 10)[0].model_dump()


# --- live filings -----------------------------------------------------------


@pytest.mark.network
def test_chunk_filing_against_a_live_filing() -> None:
    from finai.ingestion.clean import clean_items
    from finai.ingestion.fetch import filing_meta, get_latest_10k
    from finai.ingestion.parse import extract_filing_items

    filing = get_latest_10k("AAPL")
    sections = clean_items(extract_filing_items(filing)).sections

    payloads = chunk_filing(filing_meta(filing, "AAPL"), sections)

    assert {p.section for p in payloads} == set(sections)
    assert {(p.ticker, p.form, p.accession) for p in payloads} == {
        ("AAPL", "10-K", filing.accession_no)
    }
    assert {p.fiscal_year for p in payloads} == {int(str(filing.period_of_report)[:4])}
    keys = [(p.section, p.chunk_index) for p in payloads]
    assert len(keys) == len(set(keys))


@pytest.mark.network
@pytest.mark.parametrize("ticker", ["AAPL", "F", "JPM", "KO", "XOM"])
def test_chunk_live_filings(ticker: str) -> None:
    """Five filers with different table layouts: JPM splits most, XOM's units
    read "millions of dollars"."""
    from finai.ingestion.clean import clean_items
    from finai.ingestion.fetch import get_latest_10k
    from finai.ingestion.parse import extract_filing_items

    sections = clean_items(extract_filing_items(get_latest_10k(ticker))).sections
    for item, text in sections.items():
        chunks = chunk_section(text, max_tokens=1_000, overlap_tokens=100)
        assert all(count_tokens(c) <= 1_000 for c in chunks), f"{ticker} {item}"
        # Nothing is lost: every table row and every short paragraph is in a chunk.
        for block in text.split("\n\n"):
            if count_tokens(block) <= 1_000:
                for line in block.split("\n"):
                    assert any(line in c for c in chunks), (
                        f"{ticker} {item}: {line[:60]}"
                    )
