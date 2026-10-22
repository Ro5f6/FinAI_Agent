"""Deciding whether a draft answer can leave the graph (1.5).

The deterministic checks run first: numbers against tool values (1.5.4),
citations against what was retrieved (1.5.5), portfolio figures against a
recomputation (1.5.6). Any issue means ``revise`` straight away, and the
LLM critic isn't called (1.5.7): it would cost a model call to reach a
verdict the checks already reached. Only a draft that passes them goes to
the critic, which looks for what checks can't see: a claim the passages
don't support, a wrong reading of a correct figure (Haiku's P/E direction
on 1.3.20), a question left unanswered.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.messages import AnyMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from finai.agent.verification.checks import (
    Issue,
    check_citations,
    check_numbers,
    check_portfolio_math,
)


class Verdict(BaseModel):
    """Whether a draft passes, and if not, what to fix."""

    model_config = ConfigDict(frozen=True)

    status: Literal["pass", "revise"]
    issues: list[Issue] = Field(default_factory=list)
    # "checks" or "critic": which stage decided.
    by: str = "checks"


# Given the question, the draft and the state, the critic's verdict.
Critic = Callable[[str, str, dict[str, Any]], Verdict]


FEEDBACK_FLAG = "verifier_feedback"


def is_feedback(message: AnyMessage) -> bool:
    """A message the verifier added to send a draft back, not the user's."""
    return bool(message.additional_kwargs.get(FEEDBACK_FLAG))


def feedback_message(issues: Sequence[Issue]) -> HumanMessage:
    """The issues, as the planner reads them before answering again."""
    listed = "\n".join(f"- {i.claim}: {i.reason}" for i in issues)
    return HumanMessage(
        content=(
            "A review of your draft answer found these problems:\n"
            f"{listed}\n\n"
            "Answer the question again, fixing them. Use only what the tools "
            "returned, and call tools if you need more. If something can't be "
            "supported, say it isn't available rather than stating it."
        ),
        additional_kwargs={FEEDBACK_FLAG: True},
    )


def caveat(issues: Sequence[Issue]) -> str:
    """What the user is told about claims that stayed unverified (1.5.12)."""
    listed = "\n".join(f"- {i.claim}" for i in issues)
    return (
        "\n\nNote: I couldn't verify the following against my sources, so "
        f"treat them with caution:\n{listed}"
    )


def current_question(messages: Sequence[AnyMessage]) -> str:
    """The latest user message, the question the draft answers.

    The verifier's own feedback messages are skipped.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and not is_feedback(message):
            return str(message.content)
    return ""


def deterministic_issues(draft: str, state: dict[str, Any]) -> list[Issue]:
    """Every issue the three checks find, numbers first."""
    results = state.get("tool_results", [])
    question = current_question(state.get("messages", []))
    return [
        *check_numbers(draft, results, question),
        *check_citations(draft, results, state.get("retrieved_docs", [])),
        *check_portfolio_math(draft, results),
    ]


def verify(draft: str, state: dict[str, Any], critic: Critic | None = None) -> Verdict:
    """The checks' verdict if they find anything, else the critic's.

    Without a critic, a draft that passes the checks passes.
    """
    issues = deterministic_issues(draft, state)
    if issues:
        return Verdict(status="revise", issues=issues, by="checks")
    if critic is None:
        return Verdict(status="pass", by="checks")
    return critic(current_question(state.get("messages", [])), draft, state)
