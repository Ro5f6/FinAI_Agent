"""The conftest guard: unmarked tests cannot reach the network (1.3.5)."""

import socket

import httpx
import pytest
from conftest import NetworkBlockedError


def test_unmarked_test_cannot_open_a_connection() -> None:
    with pytest.raises(NetworkBlockedError, match="mark it"):
        socket.create_connection(("finnhub.io", 443), timeout=1)


def test_http_clients_are_blocked_too() -> None:
    """httpx (Finnhub, Qdrant) connects through the same socket call, and
    passes the guard's error through unwrapped, naming the test."""
    with pytest.raises(NetworkBlockedError, match="test_http_clients_are_blocked_too"):
        httpx.get("https://finnhub.io/api/v1/quote?symbol=AAPL", timeout=1)


def test_unix_sockets_stay_allowed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Local IPC never leaves the machine; only the missing path fails.

    A relative path: macOS caps AF_UNIX paths near 104 characters, and
    pytest's tmp_path is longer.
    """
    monkeypatch.chdir(tmp_path)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with pytest.raises(FileNotFoundError):
        client.connect("nothing.sock")
    client.close()
