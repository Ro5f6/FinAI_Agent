"""Tests for the planner model: the request it would send, built offline.

``ChatAnthropic._get_request_payload`` builds the exact request body without
calling the API, so these run without credits or network (the conftest guard
would fail any connection). ``-m anthropic`` runs the one live call.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from finai.agent import model as model_module
from finai.agent.model import REFUSAL_FALLBACK_BETA, make_chat_model, make_planner
from finai.config import settings


def _payload(question: str = "What's NVDA's P/E?") -> dict:
    planner = make_planner()
    return planner.bound._get_request_payload(
        [HumanMessage(question)], **planner.kwargs
    )


def test_request_carries_the_settings() -> None:
    payload = _payload()

    # The value comes from .env when set; the default is checked in test_settings.
    assert payload["model"] == settings.LLM_MODEL
    assert payload["max_tokens"] == settings.LLM_MAX_TOKENS
    assert payload["output_config"] == {"effort": settings.LLM_EFFORT}


def test_all_tools_are_bound() -> None:
    names = [t["name"] for t in _payload()["tools"]]

    assert names == ["get_live_market_data", "query_company_10k", "get_user_portfolio"]


def test_no_parameter_the_current_models_reject() -> None:
    """temperature: 400 on Sonnet 5, non-default 400 on the 5.5 models.
    Forced tool_choice: 400 on the 5.5 models."""
    payload = _payload()

    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "tool_choice" not in payload


def test_the_model_never_sees_a_user_argument() -> None:
    portfolio = next(
        t for t in _payload()["tools"] if t["name"] == "get_user_portfolio"
    )

    assert portfolio["input_schema"].get("properties", {}) == {}


def test_switching_models_is_a_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.setattr(settings, "LLM_EFFORT", "low")

    payload = _payload()

    assert payload["model"] == "claude-haiku-5-5"
    assert payload["output_config"] == {"effort": "low"}


def test_refusal_fallback_on_by_default() -> None:
    payload = _payload()

    assert payload["fallbacks"] == "default"
    assert payload["betas"] == [REFUSAL_FALLBACK_BETA]


def test_refusal_fallback_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needed on Bedrock, Vertex and Foundry, which reject the parameter."""
    monkeypatch.setattr(settings, "LLM_REFUSAL_FALLBACK", False)

    payload = _payload()

    assert "fallbacks" not in payload
    assert not payload.get("betas")


def test_api_key_is_not_exposed() -> None:
    chat = make_chat_model()

    assert settings.ANTHROPIC_API_KEY.get_secret_value() not in repr(chat)


def test_model_name_appears_only_in_settings() -> None:
    """Switching models must never need a code change."""
    import inspect

    source = inspect.getsource(model_module)

    assert "claude-" not in source


@pytest.mark.anthropic
def test_live_planner_calls_the_market_data_tool() -> None:
    """One short request (a fraction of a cent): the key, model and tool
    binding work end to end, and the model chooses the right tool."""
    reply = make_planner().invoke([HumanMessage("What is NVDA's current P/E ratio?")])

    assert isinstance(reply, AIMessage)
    calls = {c["name"]: c["args"] for c in reply.tool_calls}
    assert "get_live_market_data" in calls
    assert calls["get_live_market_data"]["ticker"].upper() == "NVDA"
