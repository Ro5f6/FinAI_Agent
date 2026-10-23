"""The FastAPI application.

    uvicorn finai.api.main:app --port 8000

``create_app`` builds the app; ``app`` is the instance uvicorn serves.

The agent graph is built once, at startup, in the ``lifespan`` handler
(2.1.4), and kept on ``app.state.graph`` for every request: building it
creates the Claude clients and compiles the graph, which is wasted work per
request. Its checkpointer is the async Redis saver, whose connection the
same handler closes at shutdown. Tests pass their own ``graph_factory``, so
they run a scripted graph without Redis or a model.
"""

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph

from finai.agent.session import app_version
from finai.api.routers import chat, health

GraphFactory = Callable[[], AbstractAsyncContextManager[CompiledStateGraph]]


@asynccontextmanager
async def agent_graph() -> AsyncGenerator[CompiledStateGraph]:
    """The real agent: Claude planner, Haiku critic, conversations in Redis."""
    from finai.agent.graph import build_graph
    from finai.agent.memory import async_redis_checkpointer

    async with async_redis_checkpointer() as checkpointer:
        yield build_graph(checkpointer=checkpointer)


def create_app(graph_factory: GraphFactory = agent_graph) -> FastAPI:
    """A new application with every router included."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with graph_factory() as graph:
            app.state.graph = graph
            yield

    app = FastAPI(
        title="FinAI-Agent",
        version=app_version(),
        description="Investment research answers from 10-K filings and market data.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
