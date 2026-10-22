"""The LLM critic: what the deterministic checks can't see (1.5.8-1.5.10).

It runs on ``VERIFIER_MODEL`` (Haiku 5.5), a different model from the
planner. No temperature is sent: the 5.x models reject non-default values.
Consistency comes from the deterministic checks, which run first, and from
structured output: ``with_structured_output(method="json_schema")`` uses
Claude's structured-output feature, which constrains the reply to the
schema. The default method, forced tool calling, isn't reliable on current
models (verified in langchain-anthropic 1.7.5's docstring and source).

Effort is set on the model (``VERIFIER_EFFORT``), which ChatAnthropic
merges with the structured-output ``output_config``; left unset, the request
carries no effort. Measured on a real answer with five planted errors, three
verdicts each (D-050): Haiku 5.5 at medium effort 18/18 right; at high 16/18;
Sonnet 5.5 15/18, stricter about inference stated as fact; Haiku 4.5 6/18.
"""

import json
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from finai.agent.prompts import CRITIC_PROMPT
from finai.agent.verification.checks import Issue
from finai.agent.verification.verifier import Critic, Verdict
from finai.config import settings

# Room for the reasoning (up to ~1,700 tokens measured at high) and the verdict.
CRITIC_MAX_TOKENS = 8_000


class CriticIssue(BaseModel):
    """One problem the critic found (1.5.9)."""

    model_config = ConfigDict(frozen=True)

    claim: str = Field(description="The claim in the draft, quoted short.")
    reason: str = Field(
        description="What is wrong with it, naming what the tool results do say."
    )


class CriticVerdict(BaseModel):
    """The critic's structured output (1.5.9)."""

    model_config = ConfigDict(frozen=True)

    status: Literal["pass", "revise"]
    issues: list[CriticIssue] = Field(default_factory=list)


def make_critic_model() -> Runnable:
    """The verifier model, constrained to return a ``CriticVerdict``."""
    model = ChatAnthropic(
        model=settings.VERIFIER_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=CRITIC_MAX_TOKENS,
        # Set on the model, not in output_config: the structured-output
        # binding sends its own output_config, which ChatAnthropic merges
        # with this field. Left unset, the request carries no effort at all.
        effort=settings.VERIFIER_EFFORT,
    )
    return model.with_structured_output(CriticVerdict, method="json_schema")


def evidence(state: dict[str, Any]) -> str:
    """What the tools returned in this thread, as the critic reads it."""
    results = [
        {"tool": r.get("tool"), "args": r.get("args"), "output": r.get("output")}
        for r in state.get("tool_results", [])
        if r.get("status") == "success"
    ]
    return json.dumps(results, indent=1, default=str)


def make_critic(model: Runnable | None = None) -> Critic:
    """A critic for ``verifier.verify`` that asks ``model``."""
    model = model or make_critic_model()

    def critic(question: str, draft: str, state: dict[str, Any]) -> Verdict:
        request = (
            f"<question>\n{question}\n</question>\n\n"
            f"<tool_results>\n{evidence(state)}\n</tool_results>\n\n"
            f"<draft>\n{draft}\n</draft>"
        )
        reply: CriticVerdict = model.invoke(
            [SystemMessage(CRITIC_PROMPT), HumanMessage(request)]
        )
        issues = [Issue(claim=i.claim, reason=i.reason) for i in reply.issues]
        # A "revise" with nothing to fix can't be acted on; an issue listed
        # under "pass" is still an issue.
        status = "revise" if issues else "pass"
        return Verdict(status=status, issues=issues, by="critic")

    return critic
