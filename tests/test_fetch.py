"""Tests for EDGAR access setup."""

import os

import pytest

from finai.config import settings
from finai.ingestion.fetch import configure_edgar_identity


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
