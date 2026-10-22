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


def current_question(messages: Sequence[AnyMessage]) -> str:
    """The latest user message: the question the draft answers."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
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
