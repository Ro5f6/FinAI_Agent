"""Tests for 10-K item extraction."""

import pytest

from finai.ingestion.fetch import UnsupportedFilingError
from finai.ingestion.parse import (
    SECTION_KEYS,
    TARGET_ITEMS,
    extract_filing_items,
    extract_items,
    missing_items,
)


class _StubDocument:
    """Stand-in for a parsed Document, keyed by get_sec_section's section names."""

    def __init__(self, content: dict[str, str | None]) -> None:
        self._content = content

    def get_sec_section(
        self, section_name: str, clean: bool = True, include_subsections: bool = True
    ) -> str | None:
        return self._content.get(section_name)


def _doc(by_item: dict[str, str | None]) -> _StubDocument:
    """Build a stub from item labels, so tests read in item terms."""
    return _StubDocument({SECTION_KEYS[k]: v for k, v in by_item.items()})


def _full_filing() -> _StubDocument:
    return _doc({item: f"text of {item}" for item in TARGET_ITEMS})


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
    partial = _doc({"Item 1": "business", "Item 1A": None, "Item 7": "   "})

    extracted = extract_items(partial)

    assert set(extracted) == {"Item 1"}
    assert "Item 1A" not in extracted  # None
    assert "Item 7" not in extracted  # whitespace only


def test_missing_items_reports_what_was_absent() -> None:
    extracted = extract_items(_doc({"Item 1": "business"}))

    assert missing_items(extracted) == ("Item 1A", "Item 7", "Item 7A", "Item 8")
    assert missing_items(extract_items(_full_filing())) == ()


def test_requesting_a_subset_extracts_only_that_subset() -> None:
    extracted = extract_items(_full_filing(), items=("Item 1A",))

    assert set(extracted) == {"Item 1A"}


def test_unsupported_filing_raises() -> None:
    with pytest.raises(UnsupportedFilingError):
        extract_items(None)


def test_extract_filing_items_parses_the_filing_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import finai.ingestion.parse as parse_module

    monkeypatch.setattr(parse_module, "filing_html", lambda _f: "<html></html>")
    monkeypatch.setattr(parse_module, "parse_html", lambda _h: _full_filing())

    assert set(extract_filing_items(object())) == set(TARGET_ITEMS)


def test_section_keys_carry_the_part_prefix() -> None:
    """ "risk_factors" and "Item 1A" both return None from get_sec_section."""
    assert SECTION_KEYS["Item 1A"] == "part_i_item_1a"
    assert all(k.startswith("part_") for k in SECTION_KEYS.values())


def test_unknown_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="Item 99"):
        extract_items(_full_filing(), items=("Item 99",))


@pytest.mark.network
@pytest.mark.parametrize("ticker", ["AAPL", "NVDA", "MSFT"])
def test_extract_items_against_live_filings(ticker: str) -> None:
    """Several companies, because AAPL alone hid a failure for every other one."""
    from finai.ingestion.fetch import get_latest_10k

    extracted = extract_filing_items(get_latest_10k(ticker))

    # Narrative items are present for every company checked; Item 8 is often
    # incorporated by reference, so it is not required.
    for required in ("Item 1", "Item 1A", "Item 7", "Item 7A"):
        assert required in extracted, f"{ticker} missing {required}"
    assert len(extracted["Item 1A"]) > len(extracted["Item 7A"])
    assert all(text.strip() for text in extracted.values())
