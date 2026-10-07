"""Cleaning extracted 10-K items before chunking.

``get_sec_section`` returns each item with the printed page furniture still in
it: page numbers, running headers, and the item heading itself. Left in, they
become chunks or chunk fragments that match nothing a user asks and split
paragraphs at page boundaries.

The furniture differs by filer, so nothing here names a company. Measured over
AAPL, F, MSFT, NVDA, GME, RIVN, JPM, KO, XOM and WMT:

* **Page footers** are a bare page number on a line of its own (nine filers) or
  ``Apple Inc. | 2025 Form 10-K | 5`` (AAPL). Every bare number checked was
  blank-isolated and increased monotonically, so they are page numbers rather
  than table cells, which sit on lines separated by single newlines.
* **Running headers** follow the footer at the top of the next page:
  ``Table of Contents`` (NVDA), ``PART II`` then ``Item 8`` (MSFT),
  ``Item 1.  Business (Continued)`` (F), ``Part I`` (JPM), the company name
  (GME, RIVN). They are recognised by position and repetition rather than by a
  list of known strings; ``find_running_headers`` has the exact tests.
* **The item heading** opens every section, within its first three non-blank
  lines, sometimes after a running header (``RIVIAN AUTOMOTIVE, INC.`` then
  ``Item 1. Business``). The item is already the chunk's ``section`` payload.
* **Vestigial sections** are cross-references to text elsewhere in the filing.
  The longest seen is 777 chars (XOM Item 8) and the shortest real section
  1,988 (MSFT Item 7A), so ``MIN_SECTION_CHARS`` sits between them.
"""

import re
from collections import Counter
from collections.abc import Iterable
from typing import NamedTuple

# Below this a section is a cross-reference, not content. Measured after
# cleaning: cross-references reach 777 chars, real sections start at 1,988.
MIN_SECTION_CHARS = 1_000

# Non-breaking (\xa0, frequent in every filer but MSFT), thin, figure and other
# Unicode spaces become ordinary spaces; zero-width characters are removed.
_SPACES = re.compile(r"[\t\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_RUN_OF_SPACES = re.compile(r" {2,}")
_RUN_OF_BLANK_LINES = re.compile(r"\n{3,}")

_PAGE_NUMBER = re.compile(r"\d{1,3}")
_PAGE_FOOTER = re.compile(r".*\bForm 10-K\b.*\|\s*\d{1,3}", re.IGNORECASE)
_ITEM_HEADING = re.compile(r"item\s*\d{1,2}[a-c]?\b", re.IGNORECASE)
# Footnote separators such as "__________" (F).
_RULE_LINE = re.compile(r"[_=*\-]{3,}")

_HEADER_WINDOW = 2  # MSFT stacks two header lines: "PART II", "Item 8".
_HEADER_MIN_REPEATS = 3
_HEADER_AT_TOP_SHARE = 0.8
_STACKED_SHARE = 0.5
_HEADING_SEARCH_LINES = 3
_SENTENCE_END = (".", "!", "?", ":", ";", '"', "”", ")")


class CleanedItems(NamedTuple):
    """Cleaned sections, and the items dropped as vestigial with their length."""

    sections: dict[str, str]
    vestigial: dict[str, int]


def _normalise_lines(text: str) -> list[str]:
    """Split into lines with Unicode spaces normalised and runs collapsed."""
    text = _ZERO_WIDTH.sub("", _SPACES.sub(" ", text))
    return [_RUN_OF_SPACES.sub(" ", line).strip() for line in text.split("\n")]


def _is_page_marker(lines: list[str], i: int) -> bool:
    """Whether ``lines[i]`` is a page footer standing alone between blank lines."""
    line = lines[i]
    if not (_PAGE_NUMBER.fullmatch(line) or _PAGE_FOOTER.fullmatch(line)):
        return False
    blank_before = i == 0 or not lines[i - 1]
    blank_after = i == len(lines) - 1 or not lines[i + 1]
    return blank_before and blank_after


def _following_nonblank(lines: list[str], i: int, n: int) -> list[int]:
    """Indexes of up to ``n`` non-blank lines after ``i``."""
    found = []
    for j in range(i + 1, len(lines)):
        if lines[j]:
            found.append(j)
            if len(found) == n:
                break
    return found


def find_running_headers(sections: Iterable[str]) -> frozenset[str]:
    """Return lines that recur at the top of pages across one filing's sections.

    Pass every section of a filing together: a header such as ``Part I`` may
    repeat only once or twice inside a short section but many times across the
    filing.

    Repetition at the top of a page is not enough on its own. Table column
    headings and subsection headings also start pages, so two further tests
    apply, both measured on the ten filers in the module docstring:

    * A header sits at the top of a page in nearly all of its occurrences.
      ``2025 2024 2023`` (AAPL) and ``Fiscal Years Ended January 31,`` (WMT)
      do so in about 20%, real headers in 86-100%.
    * A second header line, stacked under a first, appears on most of that
      header's pages in the section. ``Item 7`` under ``PART II`` (MSFT) does
      on 14 of 14 pages; ``Ford Credit Segment`` (F), a subsection that three
      times happens to start a page, on 3 of about 50.
    """
    total: Counter[str] = Counter()
    at_top: Counter[str] = Counter()
    first: Counter[str] = Counter()
    pages_under: Counter[tuple[int, str]] = Counter()
    stacked: Counter[tuple[int, str, str]] = Counter()

    for s, lines in enumerate(_normalise_lines(text) for text in sections):
        total.update(line for line in lines if line)
        for i in range(len(lines)):
            if not _is_page_marker(lines, i):
                continue
            top = [lines[j] for j in _following_nonblank(lines, i, _HEADER_WINDOW)]
            at_top.update(top)
            if top:
                first[top[0]] += 1
                pages_under[(s, top[0])] += 1
            if len(top) == _HEADER_WINDOW:
                stacked[(s, top[0], top[1])] += 1

    def mostly_at_top(line: str) -> bool:
        return at_top[line] >= _HEADER_AT_TOP_SHARE * total[line]

    headers = {
        line
        for line, n in first.items()
        if n >= _HEADER_MIN_REPEATS and mostly_at_top(line)
    }
    headers |= {
        line
        for (s, above, line), n in stacked.items()
        if above in headers
        and n >= _HEADER_MIN_REPEATS
        and n >= _STACKED_SHARE * pages_under[(s, above)]
        and mostly_at_top(line)
    }
    return frozenset(headers)


def _drop_opening_heading(lines: list[str]) -> list[str]:
    """Drop the item heading and anything printed above it.

    The search stops at the first page footer: past it, an ``Item 1A ...
    (Continued)`` running header would match and take the first page with it.
    """
    nonblank = [i for i, line in enumerate(lines) if line][:_HEADING_SEARCH_LINES]
    for i in nonblank:
        if _is_page_marker(lines, i):
            break
        if _ITEM_HEADING.match(lines[i]):
            return lines[i + 1 :]
    return lines


def clean_section(text: str, running_headers: frozenset[str] = frozenset()) -> str:
    """Return one item's text without page furniture or its opening heading.

    Paragraphs stay separated by a blank line and table rows by a single
    newline, which is the structure the token splitter in 1.2.9 splits on.
    A paragraph that a page break cut mid-sentence is rejoined.
    """
    lines = _drop_opening_heading(_normalise_lines(text))

    kept: list[str] = []
    i = 0
    while i < len(lines):
        if not _is_page_marker(lines, i):
            if not _RULE_LINE.fullmatch(lines[i]):
                kept.append(lines[i])
            i += 1
            continue

        # Skip the footer and any running header lines that open the next page.
        resume = i + 1
        for j in _following_nonblank(lines, i, _HEADER_WINDOW):
            if lines[j] not in running_headers:
                break
            resume = j + 1
        i = resume

        nxt = _following_nonblank(lines, i - 1, 1)
        prev = next((k for k in range(len(kept) - 1, -1, -1) if kept[k]), None)
        if (
            nxt
            and prev is not None
            and not kept[prev].endswith(_SENTENCE_END)
            and lines[nxt[0]][:1].islower()
        ):
            # The page break fell mid-sentence: rejoin the halves.
            kept[prev] = f"{kept[prev]} {lines[nxt[0]]}"
            del kept[prev + 1 :]
            i = nxt[0] + 1

    return _RUN_OF_BLANK_LINES.sub("\n\n", "\n".join(kept)).strip()


def clean_items(
    items: dict[str, str], min_chars: int = MIN_SECTION_CHARS
) -> CleanedItems:
    """Clean every item of one filing and drop the vestigial ones.

    Dropped items are reported rather than silently lost: when Item 7 is
    vestigial, the filing's MD&A exists but lives outside the item (JPM, XOM),
    which a caller should log rather than mistake for an absent section.
    """
    headers = find_running_headers(items.values())
    sections: dict[str, str] = {}
    vestigial: dict[str, int] = {}
    for item, text in items.items():
        cleaned = clean_section(text, headers)
        if len(cleaned) >= min_chars:
            sections[item] = cleaned
        else:
            vestigial[item] = len(cleaned)
    return CleanedItems(sections, vestigial)
