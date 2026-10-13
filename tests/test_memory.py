"""Tests for the Redis checkpointer (1.3.16). Offline tests use a stand-in
saver; ``-m redis`` runs against the Redis from docker compose."""

import uuid
from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from finai.agent import memory
from finai.agent.graph import build_graph
from finai.agent.session import answer_text, ask
from finai.config import settings


class _FakeSaver:
    instances: ClassVar[list["_FakeSaver"]] = []

    def __init__(self, redis_url=None, *, ttl=None) -> None:
        self.redis_url, self.ttl, self.setups = redis_url, ttl, 0
        _FakeSaver.instances.append(self)

    def setup(self) -> None:
        self.setups += 1


@pytest.fixture
def fake_saver(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSaver]:
    _FakeSaver.instances = []
    monkeypatch.setattr(memory, "RedisSaver", _FakeSaver)
    memory.redis_checkpointer.cache_clear()
    yield _FakeSaver
    memory.redis_checkpointer.cache_clear()


def test_ttl_is_the_setting_in_minutes() -> None:
    assert memory.ttl_config() == {
        "default_ttl": settings.CHECKPOINT_TTL_MINUTES,
        "refresh_on_read": True,
    }
    assert settings.CHECKPOINT_TTL_MINUTES == 30 * 24 * 60  # 30 days


def test_one_saver_per_process_set_up_once(fake_saver) -> None:
    first = memory.redis_checkpointer()
    second = memory.redis_checkpointer()

    assert first is second
    assert len(fake_saver.instances) == 1
    assert first.setups == 1
    assert first.redis_url == settings.REDIS_URL
    assert first.ttl == memory.ttl_config()


# --- against the compose Redis ------------------------------------------------


def _counting_model() -> RunnableLambda:
    def reply(messages, config=None):
        questions = sum(isinstance(m, HumanMessage) for m in messages)
        return AIMessage(f"I can see {questions} question(s).")

    return RunnableLambda(reply)


@pytest.fixture
def thread_id():
    """A unique thread, deleted afterwards so tests leave Redis clean."""
    tid = f"test-{uuid.uuid4()}"
    yield tid
    memory.redis_checkpointer.cache_clear()
    memory.redis_checkpointer().delete_thread(tid)
    memory.redis_checkpointer.cache_clear()


@pytest.mark.redis
def test_a_fresh_saver_continues_the_thread(thread_id: str) -> None:
    """Two savers, as two processes or two pods would have."""
    memory.redis_checkpointer.cache_clear()
    ask(
        build_graph(_counting_model(), memory.redis_checkpointer()),
        "First?",
        user_id="demo",
        thread_id=thread_id,
    )

    memory.redis_checkpointer.cache_clear()  # a new saver, nothing in memory
    state = ask(
        build_graph(_counting_model(), memory.redis_checkpointer()),
        "Second?",
        user_id="demo",
        thread_id=thread_id,
    )

    assert answer_text(state) == "I can see 2 question(s)."


@pytest.mark.redis
def test_every_saved_key_expires(thread_id: str) -> None:
    """Keys from a saver without a TTL never expire; ours must all have one."""
    import redis

    memory.redis_checkpointer.cache_clear()
    ask(
        build_graph(_counting_model(), memory.redis_checkpointer()),
        "Q",
        user_id="demo",
        thread_id=thread_id,
    )

    client = redis.Redis.from_url(settings.REDIS_URL)
    keys = list(client.scan_iter(f"*{thread_id}*"))
    ttls = {client.ttl(k) for k in keys}

    assert keys
    assert all(0 < t <= settings.CHECKPOINT_TTL_MINUTES * 60 for t in ttls), ttls


@pytest.mark.redis
def test_delete_thread_removes_it(thread_id: str) -> None:
    saver = memory.redis_checkpointer()
    graph = build_graph(_counting_model(), saver)
    ask(graph, "Q", user_id="demo", thread_id=thread_id)

    saver.delete_thread(thread_id)

    assert graph.get_state({"configurable": {"thread_id": thread_id}}).values == {}
