"""Split cleaned 10-K sections into chunks that keep tables readable.

A table row such as ``iPhone  $  209,586  4%  $  201,183`` ties each figure to
a year only by its column. A chunk that loses the heading row ``2025  Change
2024`` lets the model report a 2024 figure as 2025 (D-017), so tables are
treated as units rather than as text:

* **Table blocks** are runs of lines in which most lines contain a cell gap,
  the two spaces ``clean`` keeps between cells. Blocks separated only by a
  blank line are one table: F prints one statement as several blocks.
* **Captions** are the short paragraphs right above a table (title, unit line,
  an intro ending in ``:``). They travel with the table.
* **A table that fits** in one chunk is never split.
* **A larger table** is split between rows. Every piece repeats the caption,
  the heading rows and the current row-group label (``2024``, ``Level 1:``).

Prose is packed paragraph by paragraph. A paragraph longer than a chunk is cut
at sentence ends by LangChain's recursive splitter. Consecutive chunks share up
to ``overlap_tokens`` of whole sentences; table rows are never repeated as
overlap, since split tables already repeat their headings.

Sizes are tiktoken ``cl100k_base`` counts. voyage-4 counts 8.6% more on these
filings (1,000 here is about 1,086 there), far below its 32,000-token limit.

Measured on the latest 10-K of ten filers (AAPL, NVDA, MSFT, F, JPM, KO, GME,
RIVN, WMT, XOM); ``tests/test_chunk.py`` repeats the check.
"""

import re
from functools import lru_cache
from typing import NamedTuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finai.config import settings

ENCODING = "cl100k_base"

# A cell boundary: two spaces between non-space characters.
_CELL_GAP = re.compile(r"\S  \S")
# A financial figure: 1,234 / $ / % / 1.5 / (12 for negatives.
_FIGURE = re.compile(r"\d,\d{3}|\$|%|\d\.\d|\(\d")
# Heading rows stop at the first row holding a figure, or at this many rows.
_MAX_HEADING_ROWS = 4
# A caption is a title, a unit line or an intro sentence ending in ":".
_CAPTION_MAX_CHARS = 160
_MAX_CAPTIONS = 3
# Statement footers close the table above them. Read as captions, they would
# introduce the next statement with the previous one's footnote.
_STATEMENT_FOOTER = re.compile(
    r"(see )?(the )?accompanying notes (are an integral part|to )", re.IGNORECASE
)
_PARAGRAPH_BREAK = "\n\n"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class _Unit(NamedTuple):
    """A paragraph of prose, or a table with the captions above it."""

    is_table: bool
    text: str
    caption: str = ""
    rows: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """Tokens in ``text`` as chunk sizes measure them."""
    return len(_encoding().encode(text, disallowed_special=()))


def _is_table(block: str) -> bool:
    lines = block.split("\n")
    with_gaps = sum(bool(_CELL_GAP.search(line)) for line in lines)
    return len(lines) >= 2 and with_gaps >= len(lines) / 2


def _is_caption(paragraph: str) -> bool:
    if _STATEMENT_FOOTER.match(paragraph):
        return False
    return len(paragraph) <= _CAPTION_MAX_CHARS or paragraph.rstrip().endswith(":")


def _units(text: str) -> list[_Unit]:
    units: list[_Unit] = []
    for block in text.split(_PARAGRAPH_BREAK):
        if not block.strip():
            continue
        if not _is_table(block):
            units.append(_Unit(False, block))
            continue
        rows = tuple(block.split("\n"))
        if units and units[-1].is_table:
            # Only a blank line separates it from the table above: one table.
            prev = units.pop()
            units.append(_table(prev.caption, prev.rows + rows))
            continue
        captions: list[str] = []
        while (
            units
            and not units[-1].is_table
            and _is_caption(units[-1].text)
            and len(captions) < _MAX_CAPTIONS
        ):
            captions.insert(0, units.pop().text)
        units.append(_table(_PARAGRAPH_BREAK.join(captions), rows))
    return units


def _table(caption: str, rows: tuple[str, ...]) -> _Unit:
    return _Unit(True, _table_text(caption, rows), caption, rows)


def _table_text(caption: str, rows: tuple[str, ...] | list[str]) -> str:
    body = "\n".join(rows)
    return f"{caption}{_PARAGRAPH_BREAK}{body}" if caption else body


def _heading_rows(rows: tuple[str, ...]) -> tuple[str, ...]:
    """Leading rows before the first figure: the column headings.

    A label row such as ``Net sales:`` also has no figure, but it names the
    rows below it rather than the columns, so it ends the heading.
    """
    heading: list[str] = []
    for row in rows[:_MAX_HEADING_ROWS]:
        if _FIGURE.search(row) or row.rstrip().endswith(":"):
            break
        heading.append(row)
    return tuple(heading) if len(heading) < len(rows) else ()


def _is_group_label(row: str) -> bool:
    """A row naming a block of rows (``2024``, ``Level 1:``), not holding data."""
    return not _CELL_GAP.search(row) and not _FIGURE.search(row)


def _split_table(unit: _Unit, max_tokens: int) -> list[str]:
    """Split a table between rows; each piece repeats caption, headings, group."""
    if count_tokens(unit.text) <= max_tokens:
        return [unit.text]
    heading = _heading_rows(unit.rows)
    fixed = count_tokens(_table_text(unit.caption, heading)) + 1
    pieces: list[str] = []
    rows: list[str] = []
    used = fixed
    group: str | None = None
    for row in unit.rows[len(heading) :]:
        cost = count_tokens(row) + 1
        if rows and used + cost > max_tokens:
            pieces.append(_table_text(unit.caption, [*heading, *rows]))
            rows = (
                [f"{group} (continued)"] if group and not _is_group_label(row) else []
            )
            used = fixed + sum(count_tokens(r) + 1 for r in rows)
        if _is_group_label(row):
            group = row
        rows.append(row)
        used += cost
    if rows:
        pieces.append(_table_text(unit.caption, [*heading, *rows]))
    return pieces


def _prose_splitter(
    max_tokens: int, overlap_tokens: int
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n", ". ", "; ", ", ", " "],
        keep_separator="end",
    )


def _overlap_tail(paragraph: str, overlap_tokens: int) -> str:
    """The last whole sentences of ``paragraph`` within ``overlap_tokens``."""
    tail: list[str] = []
    used = 0
    for sentence in reversed(_SENTENCE_END.split(paragraph)):
        cost = count_tokens(sentence) + 1
        if used + cost > overlap_tokens:
            break
        tail.insert(0, sentence)
        used += cost
    return " ".join(tail)


def chunk_section(
    text: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """Split one cleaned item into chunks of at most ``max_tokens`` tokens.

    Defaults come from ``settings.CHUNK_TOKENS`` and
    ``settings.CHUNK_OVERLAP_TOKENS``.
    """
    max_tokens = max_tokens or settings.CHUNK_TOKENS
    if overlap_tokens is None:
        overlap_tokens = settings.CHUNK_OVERLAP_TOKENS
    splitter = _prose_splitter(max_tokens, overlap_tokens)

    chunks: list[str] = []
    parts: list[str] = []
    used = 0
    # Text that may open the next chunk as overlap: the last prose paragraph.
    last_prose: str | None = None
    # True while ``parts`` holds only overlap, which is never a chunk alone.
    overlap_only = False

    def flush(carry_overlap: bool) -> None:
        nonlocal used, last_prose, overlap_only
        if parts and not overlap_only:
            chunks.append(_PARAGRAPH_BREAK.join(parts))
        tail = (
            _overlap_tail(last_prose, overlap_tokens)
            if carry_overlap and last_prose
            else ""
        )
        parts.clear()
        used = 0
        last_prose = None
        overlap_only = bool(tail)
        if tail:
            parts.append(tail)
            used = count_tokens(tail)

    def add(piece: str, is_prose: bool) -> None:
        nonlocal used, last_prose, overlap_only
        cost = count_tokens(piece) + (2 if parts else 0)
        if parts and used + cost > max_tokens:
            flush(carry_overlap=True)
            cost = count_tokens(piece) + (2 if parts else 0)
            if parts and used + cost > max_tokens:
                # The overlap and this piece don't fit together; drop the overlap.
                parts.clear()
                used = 0
                cost = count_tokens(piece)
        parts.append(piece)
        used += cost
        last_prose = piece if is_prose else None
        overlap_only = False

    for unit in _units(text):
        if unit.is_table:
            pieces = _split_table(unit, max_tokens)
            if len(pieces) == 1:
                add(pieces[0], is_prose=False)
                continue
            # A split table's pieces are chunks of their own.
            flush(carry_overlap=False)
            chunks.extend(pieces)
            continue
        if count_tokens(unit.text) <= max_tokens:
            add(unit.text, is_prose=True)
            continue
        # The splitter overlaps its own pieces; no overlap is carried into them.
        flush(carry_overlap=False)
        for piece in splitter.split_text(unit.text):
            add(piece, is_prose=True)

    flush(carry_overlap=False)
    return chunks
