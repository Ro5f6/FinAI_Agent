"""Tests for EDGAR access.

The offline tests stub out ``Company`` so the default suite neither touches the
network nor consumes EDGAR rate limit. The live call is marked ``network`` and
deselected unless asked for.
"""

import os
from typing import Any

import pytest

from finai.config import settings
from finai.ingestion import fetch
from finai.ingestion.fetch import (
    FilingNotFoundError,
    configure_edgar_identity,
    get_latest_10k,
)


def test_configure_registers_identity_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)

    registered = configure_edgar_identity()

    # edgartools communicates the identity to the SEC through this variable.
    assert os.environ["EDGAR_IDENTITY"] == settings.SEC_IDENTITY
    assert registered == settings.SEC_IDENTITY


def test_configure_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetching several filings re-registers the identity; that must be safe."""
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)

    assert configure_edgar_identity() == configure_edgar_identity()


def test_identity_is_not_anonymous() -> None:
    """A blank identity is what the SEC rejects, so guard against it."""
    assert configure_edgar_identity().strip() != ""


class _StubFilings:
    def __init__(self, latest: object | None) -> None:
        self._latest = latest

    def latest(self) -> object | None:
        return self._latest


def _stub_company(
    monkeypatch: pytest.MonkeyPatch, latest: object | None
) -> dict[str, Any]:
    """Replace Company with a stub and capture the get_filings keyword arguments."""
    captured: dict[str, Any] = {}

    class _StubCompany:
        def __init__(self, ticker: str) -> None:
            captured["ticker"] = ticker

        def get_filings(self, **kwargs: Any) -> _StubFilings:
            captured.update(kwargs)
            return _StubFilings(latest)

    monkeypatch.setattr(fetch, "Company", _StubCompany)
    return captured


def test_get_latest_10k_excludes_amendments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 10-K/A must never stand in for the annual report itself."""
    sentinel = object()
    captured = _stub_company(monkeypatch, sentinel)

    assert get_latest_10k("AAPL") is sentinel
    assert captured["ticker"] == "AAPL"
    assert captured["form"] == "10-K"
    assert captured["amendments"] is False


def test_get_latest_10k_raises_when_no_filings(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty result set returns None from latest() rather than raising."""
    _stub_company(monkeypatch, None)

    with pytest.raises(FilingNotFoundError, match="NOFILINGS"):
        get_latest_10k("NOFILINGS")


@pytest.mark.network
def test_get_latest_10k_against_live_edgar() -> None:
    filing = get_latest_10k("AAPL")

    assert filing.form == "10-K"
    assert filing.accession_no
    assert filing.filing_date is not None
