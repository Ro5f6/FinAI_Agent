"""Tests for 10-K item extraction."""

import pytest

from finai.ingestion.parse import (
    TARGET_ITEMS,
    UnsupportedFilingError,
    extract_filing_items,
    extract_items,
    missing_items,
)


class _StubTenK:
    """Stand-in for TenK, exposing only the get() accessor used by the parser."""

    def __init__(self, content: dict[str, str | None]) -> None:
        self._content = content

    def get(self, item: str, default: str | None = None) -> str | None:
        return self._content.get(item, default)


def _full_filing() -> _StubTenK:
    return _StubTenK({item: f"text of {item}" for item in TARGET_ITEMS})


def test_extracts_each_target_item_separately() -> None:
    extracted = extract_items(_full_filing())

    assert set(extracted) == set(TARGET_ITEMS)
    assert extracted["Item 1A"] == "text of Item 1A"
    # Item 7 and Item 7A are distinct items, not one merged section.
    assert extracted["Item 7"] != extracted["Item 7A"]


def test_target_items_cover_the_five_the_plan_requires() -> None:
    assert TARGET_ITEMS == ("Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8")


def test_absent_items_are_omitted_not_empty() -> None:
    """An empty string would chunk to nothing and look like success."""
    partial = _StubTenK({"Item 1": "business", "Item 1A": None, "Item 7": "   "})

    extracted = extract_items(partial)

    assert set(extracted) == {"Item 1"}
    assert "Item 1A" not in extracted  # None
    assert "Item 7" not in extracted  # whitespace only


def test_missing_items_reports_what_was_absent() -> None:
    extracted = extract_items(_StubTenK({"Item 1": "business"}))

    assert missing_items(extracted) == ("Item 1A", "Item 7", "Item 7A", "Item 8")
    assert missing_items(extract_items(_full_filing())) == ()


def test_requesting_a_subset_extracts_only_that_subset() -> None:
    extracted = extract_items(_full_filing(), items=("Item 1A",))

    assert set(extracted) == {"Item 1A"}


def test_unsupported_filing_raises() -> None:
    with pytest.raises(UnsupportedFilingError):
        extract_items(None)


def test_extract_filing_items_uses_the_structured_object() -> None:
    class _StubFiling:
        def obj(self) -> _StubTenK:
            return _full_filing()

    assert set(extract_filing_items(_StubFiling())) == set(TARGET_ITEMS)


@pytest.mark.network
def test_extract_items_against_live_filing() -> None:
    from finai.ingestion.fetch import get_latest_10k

    extracted = extract_filing_items(get_latest_10k("AAPL"))

    assert missing_items(extracted) == ()
    # Risk factors are the largest narrative section by a wide margin.
    assert len(extracted["Item 1A"]) > len(extracted["Item 7"])
    assert all(text.strip() for text in extracted.values())
