"""Test-session environment setup.

Importing anything under ``finai.config`` executes the package ``__init__``,
which builds the ``Settings`` singleton. That requires the variables with no
default, so they must exist before the first test module is imported --
otherwise collection fails, with no local ``.env`` in CI.

pytest imports this file before test modules, which makes it the right place.
Individual tests still clear and set what they need through the ``clean_env``
fixture, so no test depends on these values.
"""

import os

from dotenv import dotenv_values

# Placeholders only for variables that neither the environment nor a local
# .env provides. An exported placeholder would override .env (environment
# variables win in pydantic-settings), so live tests marked `voyage` would
# send it to the API instead of the real key.
_PLACEHOLDERS = {
    "ANTHROPIC_API_KEY": "sk-ant-collection-only",
    "SEC_IDENTITY": "Collection Only collection@example.com",
    "VOYAGE_API_KEY": "pa-collection-only",
    "FINNHUB_API_KEY": "finnhub-collection-only",
}
# Tests never send LangSmith traces, whatever the shell or .env says.
os.environ["LANGSMITH_TRACING"] = "false"

_local_env = dotenv_values(".env")
for _name, _placeholder in _PLACEHOLDERS.items():
    if not os.environ.get(_name) and not _local_env.get(_name):
        os.environ[_name] = _placeholder


# --- no network outside marked tests (1.3.5) --------------------------------

import socket

import pytest

LIVE_MARKERS = {"network", "voyage", "qdrant", "finnhub", "anthropic", "redis"}


class NetworkBlockedError(RuntimeError):
    """An unmarked test tried to open a network connection."""


@pytest.fixture(autouse=True)
def _block_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail any outbound connection from a test without a live marker.

    Offline tests must pass without a network, as in CI; a hidden call to
    Finnhub, Voyage or the SEC would otherwise pass here and fail there.
    Unix-domain sockets stay allowed (they never leave the machine).
    """
    if LIVE_MARKERS & {m.name for m in request.node.iter_markers()}:
        return
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: object) -> None:
        if self.family == socket.AF_UNIX:
            return real_connect(self, address)
        raise NetworkBlockedError(
            f"{request.node.nodeid} connected to {address!r}; mark it "
            f"{' / '.join(sorted(LIVE_MARKERS))} or mock the call"
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect)
