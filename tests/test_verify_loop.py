"""The verify loop end to end, with a scripted planner (1.5.13, 1.5.16, 1.5.17).

The tools are real (the quote is stubbed); the planner is a script that
answers wrongly a given number of times; the verifier is the real one, with
the deterministic checks and, where a test passes one, a scripted critic.
"""

import datetime as dt

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from finai.agent.graph import build_graph
from finai.agent.session import answer_text, ask
from finai.agent.verification.checks import Issue
from finai.agent.verification.verifier import Verdict, current_question, is_feedback
from finai.config import settings
from finai.tools import market_data

RIGHT = "NVDA is at $212.21. [Finnhub quote, 2026-10-22 20:00 UTC]"
WRONG = "NVDA is at $221.21. [Finnhub quote, 2026-10-22 20:00 UTC]"  # digits swapped


@pytest.fixture(autouse=True)
def quote(monkeypatch: pytest.MonkeyPatch) -> None:
    data = market_data.MarketData(
        ticker="NVDA", name="NVIDIA", price=212.21, price_currency="USD", change_pct=-4.4,
        market_cap=None, market_cap_currency=None, week_52_high=None, week_52_low=None,
        pe_ttm=None, pe_annual=None, pe_source=None, eps_ttm=None, eps_ttm_period_end=None,
        eps_annual=None, eps_annual_period_end=None, eps_currency=None, eps_per=None,
        quote_time=dt.datetime(2026, 10, 22, 20, 0, tzinfo=dt.UTC),
    )  # fmt: skip
    monkeypatch.setattr(market_data, "get_live_market_data", lambda ticker: data)


class Script:
    """Asks for the quote once, then answers wrongly ``wrong`` times, then right."""

    def __init__(self, wrong: int) -> None:
        self.wrong = wrong
        self.drafts: list[str] = []
        self.feedback_seen: list[str] = []

    def __call__(self, messages, config=None) -> AIMessage:
        if not any(getattr(m, "tool_calls", None) for m in messages):
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "q1",
                        "name": "get_live_market_data",
                        "args": {"ticker": "NVDA"},
                    }
                ],
            )
        last = messages[-1]
        if isinstance(last, HumanMessage) and is_feedback(last):
            self.feedback_seen.append(last.content)
        draft = WRONG if len(self.drafts) < self.wrong else RIGHT
        self.drafts.append(draft)
        return AIMessage(draft)


def _ask(script: Script, critic=None) -> dict:
    graph = build_graph(RunnableLambda(script), critic=critic)
    return ask(graph, "What's NVDA trading at?", user_id="demo", thread_id="t")


def test_a_right_answer_passes_first_time() -> None:
    script = Script(wrong=0)
    state = _ask(script)
    assert answer_text(state) == RIGHT
    assert state["verdict"]["status"] == "pass"
    assert state["revision_count"] == 0
    assert len(script.drafts) == 1


def test_a_wrong_number_is_caught_and_corrected() -> None:
    """1.5.16: the planted wrong number goes back; the second draft passes."""
    script = Script(wrong=1)
    state = _ask(script)

    assert script.drafts == [WRONG, RIGHT]
    assert answer_text(state) == RIGHT
    assert (state["verdict"]["status"], state["revision_count"]) == ("pass", 1)
    # The planner was told what was wrong, with the value the tool returned.
    [feedback] = script.feedback_seen
    assert "$221.21 is not in any tool result" in feedback
    assert "212.21" in feedback
    # The feedback stays in the thread but is never taken for the question.
    assert sum(is_feedback(m) for m in state["messages"]) == 1
    assert current_question(state["messages"]) == "What's NVDA trading at?"


def test_an_always_wrong_planner_stops_with_a_caveat() -> None:
    """1.5.17: MAX_REVISIONS revisions, then the answer goes out flagged."""
    script = Script(wrong=99)
    state = _ask(script)

    assert len(script.drafts) == settings.MAX_REVISIONS + 1
    assert state["revision_count"] == settings.MAX_REVISIONS
    assert state["verdict"]["status"] == "revise"
    answer = answer_text(state)
    assert answer.startswith(WRONG)
    assert "I couldn't verify the following" in answer
    assert "- $221.21" in answer
    # The caveat replaced the last draft rather than adding another message.
    finals = [
        m for m in state["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    assert finals[-1].text == answer
    assert len(finals) == settings.MAX_REVISIONS + 1


def test_the_step_limit_caps_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """1.5.13: revisions can't take the run past MAX_AGENT_LOOPS."""
    monkeypatch.setattr(settings, "MAX_REVISIONS", 10)
    monkeypatch.setattr(settings, "MAX_AGENT_LOOPS", 3)
    script = Script(wrong=99)
    state = _ask(script)

    assert state["loop_count"] == 3  # 1 tool request + 2 drafts
    assert len(script.drafts) == 2
    assert "I couldn't verify the following" in answer_text(state)


def test_the_critic_judges_what_the_checks_pass() -> None:
    """A draft with correct figures can still be sent back by the critic."""
    verdicts = iter(
        [
            Verdict(
                status="revise",
                issues=[Issue(claim="c", reason="unsupported")],
                by="critic",
            ),
            Verdict(status="pass", by="critic"),
        ]
    )
    asked: list[str] = []

    def critic(question, draft, state) -> Verdict:
        asked.append(question)
        return next(verdicts)

    state = _ask(Script(wrong=0), critic=critic)

    assert asked == ["What's NVDA trading at?"] * 2  # never the feedback
    assert (state["verdict"]["by"], state["revision_count"]) == ("critic", 1)


def test_a_failed_check_never_reaches_the_critic() -> None:
    """1.5.7 in the graph: wrong drafts stop at the checks, at no model cost."""
    calls: list[str] = []

    def critic(question, draft, state) -> Verdict:
        calls.append(draft)
        return Verdict(status="pass", by="critic")

    state = _ask(Script(wrong=99), critic=critic)

    assert calls == []
    assert state["verdict"]["by"] == "checks"


def test_each_question_starts_with_fresh_revisions() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    script = Script(wrong=1)
    graph = build_graph(
        RunnableLambda(script), checkpointer=InMemorySaver(), critic=None
    )
    ask(graph, "What's NVDA trading at?", user_id="demo", thread_id="t")
    script.wrong = 0
    state = ask(graph, "And now?", user_id="demo", thread_id="t")
    assert (state["revision_count"], state["verdict"]["status"]) == (0, "pass")
