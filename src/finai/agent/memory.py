"""Conversation memory in Redis (1.3.16).

The graph's checkpointer saves a thread's state after every step under its
``thread_id``. In process memory it would vanish on restart and differ between
the 2-4 API pods; Redis is shared by every pod and outlives them.

Verified against langgraph-checkpoint-redis 0.5.2 on the compose Redis 8
(docs/CONTEXT.md):

* ``setup()`` creates two search indexes (``checkpoint``, ``checkpoint_write``)
  and is safe to call again.
* A fresh saver continues a thread another saver wrote, as after a restart.
* ``ttl={"default_ttl": N}`` is in **minutes** (the library multiplies by 60).
  Every saver must get it: keys written by a saver without it never expire.
  That's why savers are only made here, from one setting.
* The API runs the graph with ``ainvoke``/``astream``, which need the async
  ``AsyncRedisSaver``: the sync ``RedisSaver`` raises ``NotImplementedError``
  there. Entering ``AsyncRedisSaver.from_conn_string`` creates the indexes
  (``__aenter__`` calls ``asetup``); leaving it closes the connection.
* ``RedisSaver`` keeps every step (23 keys, ~36 KB for two short turns);
  ``ShallowRedisSaver`` keeps only the latest state (14 keys, ~18 KB). The full
  history is kept while the agent is being built, for ``get_state_history``.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from finai.config import settings


def ttl_config() -> dict[str, object]:
    """Expire threads after the configured minutes; reading one renews it."""
    return {"default_ttl": settings.CHECKPOINT_TTL_MINUTES, "refresh_on_read": True}


@lru_cache(maxsize=1)
def redis_checkpointer() -> RedisSaver:
    """One saver per process, connected to ``REDIS_URL``, indexes created."""
    saver = RedisSaver(redis_url=settings.REDIS_URL, ttl=ttl_config())
    saver.setup()
    return saver


@asynccontextmanager
async def async_redis_checkpointer() -> AsyncGenerator[AsyncRedisSaver]:
    """The async saver for the API, with the same TTL; closed on exit."""
    async with AsyncRedisSaver.from_conn_string(
        settings.REDIS_URL, ttl=ttl_config()
    ) as saver:
        yield saver
