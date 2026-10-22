"""Tests for the LLM critic (1.5.8-1.5.10).

Offline tests replace the model; ``-m anthropic`` asks Haiku 5.5 about a
small draft with a planted reasoning error, the kind no check can see.
"""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from finai.agent.prompts import CRITIC_PROMPT
from finai.agent.verification.critic import (
    CriticIssue,
    CriticVerdict,
    evidence,
    make_critic,
    make_critic_model,
)
from finai.config import settings

QUOTE = {
    "tool": "get_live_market_data",
    "args": {"ticker": "NVDA"},
    "status": "success",
    "output": {"price": 212.21, "pe_ttm": 26.85, "pe_annual": 43.37, "eps_ttm": 7.9108,
               "eps_annual": 4.8979, "quote_time": "2026-10-22T20:00:00Z"},
}  # fmt: skip
STATE = {
    "messages": [
        HumanMessage("How does NVDA's trailing P/E compare with last year's?")
    ],
    "tool_results": [
        QUOTE,
        {
            "tool": "query_company_10k",
            "args": {},
            "status": "error",
            "output": "bad section",
        },
    ],
}


def _model(reply: CriticVerdict, seen: list):
    def answer(messages):
        seen.append(messages)
        return reply

    return RunnableLambda(answer)


def test_the_critic_sends_its_prompt_and_the_evidence() -> None:
    seen: list = []
    critic = make_critic(_model(CriticVerdict(status="pass"), seen))

    verdict = critic("Q?", "The draft.", STATE)

    assert (verdict.status, verdict.by) == ("pass", "critic")
    [[system, request]] = seen
    assert isinstance(system, SystemMessage) and system.content == CRITIC_PROMPT
    assert "<question>\nQ?\n</question>" in request.content
    assert "<draft>\nThe draft.\n</draft>" in request.content
    assert '"pe_annual": 43.37' in request.content


def test_failed_tool_calls_are_not_evidence() -> None:
    text = evidence(STATE)
    assert "get_live_market_data" in text
    assert "bad section" not in text


def test_issues_decide_the_status() -> None:
    """An issue under "pass" still needs fixing; "revise" with none can't be acted on."""
    issue = CriticIssue(claim="c", reason="r")
    flagged = make_critic(_model(CriticVerdict(status="pass", issues=[issue]), []))
    empty = make_critic(_model(CriticVerdict(status="revise"), []))

    assert flagged("Q", "D", STATE).status == "revise"
    assert flagged("Q", "D", STATE).issues[0].claim == "c"
    assert empty("Q", "D", STATE).status == "pass"


def test_prompt_covers_support_reasoning_and_the_question() -> None:
    for topic in ("Support.", "Reasoning about figures.", "The question."):
        assert topic in CRITIC_PROMPT
    assert "don't recompute numbers" in CRITIC_PROMPT  # the checks did


def test_the_model_sends_effort_alongside_the_schema() -> None:
    """with_structured_output binds its own output_config; ChatAnthropic merges
    the model's effort into it (langchain-anthropic 1.7.5)."""
    bound = make_critic_model().first
    payload = bound.bound._get_request_payload([HumanMessage("hi")], **bound.kwargs)
    assert payload["output_config"]["effort"] == settings.VERIFIER_EFFORT
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["model"] == settings.VERIFIER_MODEL
    assert "temperature" not in payload


@pytest.mark.anthropic
def test_live_critic_catches_a_reversed_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "VERIFIER_MODEL", "claude-haiku-5-5")
    critic = make_critic()
    draft = (
        "NVDA's trailing P/E of 26.85 is above its fiscal-year P/E of 43.37, "
        "because earnings fell. [Finnhub quote, 2026-10-22 20:00 UTC]"
    )
    verdict = critic(STATE["messages"][0].content, draft, STATE)
    assert verdict.status == "revise"
