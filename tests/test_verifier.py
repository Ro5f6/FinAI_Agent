"""Tests for the verifier's decision: checks first, the critic only after."""

from langchain_core.messages import AIMessage, HumanMessage

from finai.agent.verification.checks import Issue
from finai.agent.verification.verifier import Verdict, current_question, verify

QUOTE = {
    "tool": "get_live_market_data",
    "status": "success",
    "output": {"price": 212.21, "quote_time": "2026-10-22T20:00:00Z"},
}
STATE = {
    "messages": [
        HumanMessage("Earlier?"),
        AIMessage("..."),
        HumanMessage("What's NVDA at?"),
    ],
    "tool_results": [QUOTE],
    "retrieved_docs": [],
}
GOOD = "NVDA is at $212.21 [Finnhub quote, 2026-10-22 20:00 UTC]."


class _Critic:
    def __init__(self, verdict: Verdict) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []

    def __call__(self, question, draft, state) -> Verdict:
        self.calls.append((question, draft))
        return self.verdict


def test_a_failed_check_revises_without_calling_the_critic() -> None:
    critic = _Critic(Verdict(status="pass", by="critic"))

    verdict = verify(
        "NVDA is at $221.21 [Finnhub quote, 2026-10-22 19:00 UTC].", STATE, critic
    )

    assert verdict.status == "revise"
    assert verdict.by == "checks"
    assert [i.claim for i in verdict.issues] == [
        "$221.21",  # numbers first
        "[Finnhub quote, 2026-10-22 19:00 UTC]",
    ]
    assert critic.calls == []  # 1.5.7: the call is saved


def test_a_draft_that_passes_the_checks_goes_to_the_critic() -> None:
    critic = _Critic(
        Verdict(
            status="revise",
            issues=[Issue(claim="x", reason="unsupported")],
            by="critic",
        )
    )

    verdict = verify(GOOD, STATE, critic)

    assert critic.calls == [("What's NVDA at?", GOOD)]  # the current question
    assert (verdict.status, verdict.by) == ("revise", "critic")


def test_without_a_critic_passing_the_checks_passes() -> None:
    assert verify(GOOD, STATE) == Verdict(status="pass", by="checks")


def test_current_question_is_the_latest_user_message() -> None:
    assert current_question(STATE["messages"]) == "What's NVDA at?"
    assert current_question([]) == ""


def test_feedback_tells_the_planner_the_user_never_saw_the_draft() -> None:
    """Live, Sonnet opened its corrected answer with "I made a typo in my first
    answer", though the user only sees the final answer."""
    from finai.agent.verification.verifier import feedback_message, is_feedback

    message = feedback_message(
        [Issue(claim="$12.212", reason="not in any tool result")]
    )
    assert is_feedback(message)
    assert "- $12.212: not in any tool result" in message.content
    assert "never saw the draft" in message.content
