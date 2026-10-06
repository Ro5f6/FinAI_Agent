"""Extracting individual items from a 10-K.

The plan suggested ``filing.obj().get("Item 1A")``. That works for AAPL and
fails for most other companies: ``TenK`` reads ``filing.html()``, which returns
None for inline-XBRL filings whose homepage lists no primary document, and the
item lookup then raises ``TypeError`` from deep inside edgartools. NVDA, MSFT,
GME, RIVN and F all fail that way.

So items are read from the parsed document instead -- still edgartools doing the
parsing, via ``parse_html`` and ``get_sec_section`` -- over the HTML that
``fetch.filing_html`` obtains reliably. Verified to return byte-identical
lengths to ``TenK.get`` for AAPL, while also working for the companies where
``TenK`` does not.

Item numbers are the ``section`` value stored on each chunk (1.2.10) and
filtered on at query time (1.3.2), so they stay verbatim as the dict keys.
"""

from edgar.documents import parse_html
from edgar.entity import EntityFiling

from finai.ingestion.fetch import UnsupportedFilingError, filing_html

# Item label -> the key get_sec_section expects. The part prefix matters:
# "risk_factors" and "Item 1A" both return None.
SECTION_KEYS: dict[str, str] = {
    "Item 1": "part_i_item_1",
    "Item 1A": "part_i_item_1a",
    "Item 7": "part_ii_item_7",
    "Item 7A": "part_ii_item_7a",
    "Item 8": "part_ii_item_8",
}

# The business description, risk factors, management's discussion, market-risk
# disclosures and the financial statements.
TARGET_ITEMS: tuple[str, ...] = tuple(SECTION_KEYS)


def extract_items(
    document: object, items: tuple[str, ...] = TARGET_ITEMS
) -> dict[str, str]:
    """Return ``{item: text}`` for each requested item that has content.

    Items absent from the filing are omitted rather than returned empty, because
    an empty string would be chunked into nothing while looking like a
    successful extraction. Call ``missing_items`` to see what was not provided.

    Length is not checked, and some items are present but vestigial: Item 8 is
    frequently a one-line cross-reference to a separate financial-statements
    index rather than the statements themselves (NVDA 207 chars, F 436 chars,
    against AAPL 62,223 and MSFT 105,683). Those pass the non-blank test and
    would become a useless chunk, so chunking should apply a minimum length.
    """
    if document is None or not hasattr(document, "get_sec_section"):
        raise UnsupportedFilingError(
            f"expected a parsed document with .get_sec_section(), got {type(document).__name__}"
        )

    unknown = set(items) - set(SECTION_KEYS)
    if unknown:
        raise ValueError(
            f"unknown items {sorted(unknown)}; known: {sorted(SECTION_KEYS)}"
        )

    extracted: dict[str, str] = {}
    for item in items:
        text = document.get_sec_section(SECTION_KEYS[item])
        if text and text.strip():
            extracted[item] = text
    return extracted


def extract_filing_items(
    filing: EntityFiling, items: tuple[str, ...] = TARGET_ITEMS
) -> dict[str, str]:
    """Extract items from a filing, parsing its HTML first."""
    return extract_items(parse_html(filing_html(filing)), items)


def missing_items(
    extracted: dict[str, str], items: tuple[str, ...] = TARGET_ITEMS
) -> tuple[str, ...]:
    """Return the requested items that the filing did not provide."""
    return tuple(item for item in items if item not in extracted)
