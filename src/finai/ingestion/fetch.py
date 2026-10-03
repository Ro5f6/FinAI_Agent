"""EDGAR access for SEC filings.

The SEC rejects anonymous requests, so an identity must be registered before
any EDGAR call. edgartools stores it in the ``EDGAR_IDENTITY`` environment
variable, which makes registration process-global and safe to repeat.
"""

from edgar import get_identity, set_identity

from finai.config import settings


def configure_edgar_identity() -> str:
    """Register the SEC contact identity for this process.

    Returns the identity edgartools will actually send, read back rather than
    echoed, so a caller sees what was registered and not merely what was asked
    for.
    """
    set_identity(settings.SEC_IDENTITY)
    return get_identity()
