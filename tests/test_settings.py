"""Tests for the Settings class and the config singleton."""

import importlib

import pytest
from pydantic import ValidationError

from finai.config.settings import Settings

# Required fields have no default, so every test must supply them.
REQUIRED_ENV = {
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "SEC_IDENTITY": "Test User test@example.com",
    "VOYAGE_API_KEY": "pa-test",
    "FINNHUB_API_KEY": "finnhub-test",
}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every Settings variable from the environment.

    Without this, a real key exported in the developer's shell would satisfy a
    test that is supposed to be checking the missing-variable path.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_loads_values_from_env(clean_env: pytest.MonkeyPatch) -> None:
    for name, value in REQUIRED_ENV.items():
        clean_env.setenv(name, value)
    clean_env.setenv("QDRANT_URL", "http://qdrant.test:6333")
    clean_env.setenv("MAX_AGENT_LOOPS", "9")

    # _env_file=None ignores any .env on disk, so the test reads only what it set.
    settings = Settings(_env_file=None)

    assert settings.QDRANT_URL == "http://qdrant.test:6333"
    assert settings.MAX_AGENT_LOOPS == 9
    assert settings.SEC_IDENTITY == REQUIRED_ENV["SEC_IDENTITY"]


def test_applies_defaults_when_unset(clean_env: pytest.MonkeyPatch) -> None:
    for name, value in REQUIRED_ENV.items():
        clean_env.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.QDRANT_URL == "http://localhost:6333"
    assert settings.REDIS_URL == "redis://localhost:6379"
    assert settings.COLLECTION_NAME == "sec_filings"
    assert settings.EMBEDDING_MODEL == "voyage-4"
    assert settings.EMBEDDING_DIM == 1024
    assert settings.CHUNK_TOKENS == 1_000
    assert settings.CHUNK_OVERLAP_TOKENS < settings.CHUNK_TOKENS
    assert settings.CHUNK_TOKENS < settings.EMBEDDING_MAX_TOKENS
    assert settings.LLM_MODEL == "claude-sonnet-5"
    assert settings.MAX_AGENT_LOOPS == 6
    assert settings.LANGSMITH_TRACING is False
    assert settings.LANGSMITH_API_KEY is None


def test_missing_required_variables_raise(clean_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    missing = {error["loc"][0] for error in exc_info.value.errors()}
    assert missing == set(REQUIRED_ENV)


def test_extra_env_vars_are_ignored(clean_env: pytest.MonkeyPatch) -> None:
    """Later phases add variables to .env before the fields exist here."""
    for name, value in REQUIRED_ENV.items():
        clean_env.setenv(name, value)
    clean_env.setenv("DATABASE_URL", "sqlite:///ignored.db")
    clean_env.setenv("MAX_REVISIONS", "2")

    settings = Settings(_env_file=None)

    assert not hasattr(settings, "DATABASE_URL")


def test_api_keys_are_not_exposed_in_repr(clean_env: pytest.MonkeyPatch) -> None:
    for name, value in REQUIRED_ENV.items():
        clean_env.setenv(name, value)

    settings = Settings(_env_file=None)

    assert "sk-ant-test" not in repr(settings)
    assert "pa-test" not in repr(settings)
    assert "finnhub-test" not in repr(settings)
    assert settings.ANTHROPIC_API_KEY.get_secret_value() == "sk-ant-test"


def test_singleton_is_importable_from_package(clean_env: pytest.MonkeyPatch) -> None:
    """The import path the application uses: `from finai.config import settings`."""
    for name, value in REQUIRED_ENV.items():
        clean_env.setenv(name, value)
    clean_env.setenv("QDRANT_URL", "http://singleton.test:6333")

    import finai.config

    # The module may already be imported with a different environment.
    module = importlib.reload(finai.config)

    assert module.settings.QDRANT_URL == "http://singleton.test:6333"
