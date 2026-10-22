"""Tests for tracing: settings reach LangSmith, runs carry metadata, verdicts
are recorded (1.4.2, 1.4.3, 1.5.18). No network: LangSmith is never called."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from finai.agent import nodes
from finai.agent.session import app_version, thread_config
from finai.agent.verification.checks import Issue
from finai.agent.verification.verifier import Verdict
from finai.config.tracing import export_tracing


def _settings(tracing=True, key="lsv2_pt_x", endpoint=None, project="finai-agent-dev"):
    return SimpleNamespace(
        LANGSMITH_TRACING=tracing,
        LANGSMITH_API_KEY=SecretStr(key) if key else None,
        LANGSMITH_PROJECT=project,
        LANGSMITH_ENDPOINT=endpoint,
    )


def test_enabled_tracing_is_exported_with_its_region() -> None:
    environ: dict[str, str] = {}
    endpoint = "https://aws.api.smith.langchain.com"
    assert export_tracing(_settings(endpoint=endpoint), environ) is True
    assert environ == {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "lsv2_pt_x",
        "LANGSMITH_PROJECT": "finai-agent-dev",
        "LANGSMITH_ENDPOINT": endpoint,
    }


def test_without_an_endpoint_a_stale_one_is_removed() -> None:
    environ = {"LANGSMITH_ENDPOINT": "https://old.example"}
    export_tracing(_settings(endpoint=None), environ)
    assert "LANGSMITH_ENDPOINT" not in environ


@pytest.mark.parametrize("settings", [_settings(tracing=False), _settings(key=None)])
def test_tracing_is_off_without_the_switch_or_a_key(settings) -> None:
    """Off is exported explicitly, so a value left in the shell can't turn it on."""
    environ = {"LANGSMITH_TRACING": "true"}
    assert export_tracing(settings, environ) is False
    assert environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in environ


def test_tests_never_trace() -> None:
    """conftest forces tracing off before the settings load."""
    import os

    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_every_run_carries_user_thread_and_version() -> None:
    config = thread_config(user_id="demo", thread_id="t-1")
    assert config["metadata"] == {
        "user_id": "demo",
        "thread_id": "t-1",
        "app_version": app_version(),
    }
    assert app_version() == "0.1.0"


class _Run:
    def __init__(self) -> None:
        self.metadata: dict = {}

    def add_metadata(self, metadata: dict) -> None:
        self.metadata.update(metadata)


def test_the_verdict_is_recorded_on_the_verifier_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _Run()
    monkeypatch.setattr(nodes, "get_current_run_tree", lambda: run)
    verdict = Verdict(
        status="revise", issues=[Issue(claim="a", reason="b")] * 2, by="critic"
    )

    nodes._record_verdict(verdict, revision=1)

    assert run.metadata == {
        "verdict_status": "revise",
        "verdict_by": "critic",
        "verdict_issues": 2,
        "revision": 1,
    }


def test_recording_a_verdict_without_tracing_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nodes, "get_current_run_tree", lambda: None)
    nodes._record_verdict(Verdict(status="pass"), revision=0)  # no error
