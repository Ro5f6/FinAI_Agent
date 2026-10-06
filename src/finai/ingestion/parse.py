"""Extracting individual items from a 10-K.

``TenK`` exposes named properties for some items (``business``,
``risk_factors``, ``management_discussion``) but not for Items 7A or 8, so
``get("Item 7A")`` is used throughout for uniformity. The named properties were
checked to return identical text, so nothing is lost by preferring ``get``.

Item numbers double as the ``section`` value stored on each chunk (1.2.10) and
filtered on at query time (1.3.2), so they are kept verbatim rather than
renamed to friendlier labels.
"""

from edgar.entity import EntityFiling

# Items worth indexing: the business description, the risk factors, management's
# discussion, market-risk disclosures and the financial statements.
TARGET_ITEMS: tuple[str, ...] = ("Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8")


class UnsupportedFilingError(TypeError):
    """The filing produced no structured object to extract items from."""


def extract_items(
    tenk: object, items: tuple[str, ...] = TARGET_ITEMS
) -> dict[str, str]:
    """Return ``{item: text}`` for each requested item that has content.

    Items absent from the filing are omitted rather than returned empty. Filings
    sometimes incorporate an item by reference or leave it out, and an empty
    string would otherwise be chunked into nothing and look like a successful
    extraction. Callers can compare against ``items`` to see what was missing.
    """
    if tenk is None or not hasattr(tenk, "get"):
        raise UnsupportedFilingError(
            f"expected a structured 10-K object with .get(), got {type(tenk).__name__}"
        )

    extracted: dict[str, str] = {}
    for item in items:
        text = tenk.get(item)
        if text and text.strip():
            extracted[item] = text
    return extracted


def extract_filing_items(
    filing: EntityFiling, items: tuple[str, ...] = TARGET_ITEMS
) -> dict[str, str]:
    """Extract items straight from a filing, via its structured object."""
    return extract_items(filing.obj(), items)


def missing_items(
    extracted: dict[str, str], items: tuple[str, ...] = TARGET_ITEMS
) -> tuple[str, ...]:
    """Return the requested items that the filing did not provide."""
    return tuple(item for item in items if item not in extracted)
