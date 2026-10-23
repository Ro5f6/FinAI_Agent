"""The FastAPI application.

    uvicorn finai.api.main:app --port 8000

``create_app`` builds the app; ``app`` is the instance uvicorn serves. The
agent graph is built once at startup (2.1.4) and the routes stream its
answers (2.1.6); this module only assembles the pieces.
"""

from fastapi import FastAPI

from finai.agent.session import app_version
from finai.api.routers import chat


def create_app() -> FastAPI:
    """A new application with every router included."""
    app = FastAPI(
        title="FinAI-Agent",
        version=app_version(),
        description="Investment research answers from 10-K filings and market data.",
    )
    app.include_router(chat.router)
    return app


app = create_app()
