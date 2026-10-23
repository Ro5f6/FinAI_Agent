"""Tests for the HTTP API."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from finai.agent.session import app_version
from finai.api.main import app, create_app


def test_the_app_assembles_with_the_chat_router() -> None:
    created = create_app()
    assert isinstance(created, FastAPI)
    assert created.version == app_version()
    assert isinstance(app, FastAPI)


def test_the_openapi_schema_is_served() -> None:
    response = TestClient(create_app()).get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "FinAI-Agent"
