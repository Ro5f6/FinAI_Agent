"""Tests for the planner node, with a stand-in model that records its input."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from finai.agent.nodes import make_planner_node
from finai.agent.prompts import SYSTEM_PROMPT
from finai.agent.state import turn_input


class _Recorder:
    """A stand-in model: records what it receives and replies with a message."""

    def __init__(self, reply: AIMessage) -> None:
        self.reply = reply
        self.inputs: list[list] = []
        self.configs: list[RunnableConfig] = []

    def runnable(self) -> RunnableLambda:
        def call(messages, config: RunnableConfig):
            self.inputs.append(messages)
            self.configs.append(config)
            return self.reply

        return RunnableLambda(call)


def _state(**overrides) -> dict:
    return {
        **turn_input("What's NVDA's P/E?", "demo"),
        "retrieved_docs": [],
        **overrides,
    }


def test_planner_sends_the_system_prompt_then_the_conversation() -> None:
    model = _Recorder(AIMessage("NVDA trades at a P/E of 29.0."))
    state = _state()

    make_planner_node(model.runnable())(state, {})

    [sent] = model.inputs
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == SYSTEM_PROMPT
    assert sent[1:] == state["messages"]


def test_planner_returns_only_the_new_reply_and_counts_the_step() -> None:
    reply = AIMessage("Answer.")
    update = make_planner_node(_Recorder(reply).runnable())(_state(loop_count=2), {})

    assert update == {"messages": [reply], "loop_count": 3}


def test_system_prompt_is_never_stored_in_the_state() -> None:
    update = make_planner_node(_Recorder(AIMessage("Answer.")).runnable())(_state(), {})

    assert not any(isinstance(m, SystemMessage) for m in update["messages"])


def test_tool_loop_history_is_passed_back_unchanged() -> None:
    """Thinking blocks and tool calls must reach the model exactly as returned."""
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    call = AIMessage(
        content=[
            thinking,
            {
                "type": "tool_use",
                "id": "t1",
                "name": "get_live_market_data",
                "input": {"ticker": "NVDA"},
            },
        ],
        tool_calls=[
            {"id": "t1", "name": "get_live_market_data", "args": {"ticker": "NVDA"}}
        ],
    )
    result = ToolMessage(content='{"status": "ok"}', tool_call_id="t1")
    state = _state(messages=[HumanMessage("NVDA P/E?"), call, result])
    model = _Recorder(AIMessage("29.0"))

    make_planner_node(model.runnable())(state, {})

    assert model.inputs[0][2] is call
    assert model.inputs[0][2].content[0] == thinking
    assert model.inputs[0][3] is result


def test_run_config_reaches_the_model() -> None:
    """Callbacks and tracing hang off the config."""
    model = _Recorder(AIMessage("Answer."))

    make_planner_node(model.runnable())(_state(), {"tags": ["eval-run"]})

    assert "eval-run" in model.configs[0]["tags"]


def test_planner_works_with_the_real_tool_bound_model_offline() -> None:
    """The real planner builds a valid request from a node call (no API call)."""
    from finai.agent.model import make_planner

    planner = make_planner()
    captured = {}

    def capture(messages, config):
        captured["payload"] = planner.bound._get_request_payload(
            messages, **planner.kwargs
        )
        return AIMessage("ok")

    make_planner_node(RunnableLambda(capture))(_state(), {})

    payload = captured["payload"]
    assert payload["system"] == SYSTEM_PROMPT
    assert [m["role"] for m in payload["messages"]] == ["user"]
    assert len(payload["tools"]) == 3
