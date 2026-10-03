"""EDGAR access for SEC filings.

The SEC rejects anonymous requests, so an identity must be registered before
any EDGAR call. edgartools stores it in the ``EDGAR_IDENTITY`` environment
variable, which makes registration process-global and safe to repeat.
"""

from edgar import Company, get_identity, set_identity
from edgar.entity import EntityFiling

from finai.config import settings


class FilingNotFoundError(LookupError):
    """No filing of the requested form exists for the requested company."""


def configure_edgar_identity() -> str:
    """Register the SEC contact identity for this process.

    Returns the identity edgartools will actually send, read back rather than
    echoed, so a caller sees what was registered and not merely what was asked
    for.
    """
    set_identity(settings.SEC_IDENTITY)
    return get_identity()


def get_latest_10k(ticker: str) -> EntityFiling:
    """Return the most recent original 10-K for ``ticker``.

    Amendments are excluded deliberately. ``get_filings`` defaults to
    ``amendments=True``, which folds 10-K/A filings into the same result set --
    AAPL has two -- so the newest entry can be an amendment rather than the
    annual report itself.

    Raises:
        FilingNotFoundError: the company exists but has filed no 10-K.
        CompanyNotFoundError: edgartools does not recognise the ticker.
    """
    configure_edgar_identity()

    filings = Company(ticker).get_filings(form="10-K", amendments=False)
    latest = filings.latest()
    if latest is None:
        raise FilingNotFoundError(f"no 10-K filings found for ticker {ticker!r}")
    return latest
