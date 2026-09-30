"""Application configuration.

Every configuration value the app reads is declared here. Nothing else calls
``os.getenv``; import the ``settings`` singleton instead.

Field names are uppercase so they map one-to-one onto the variable names in
``.env.example`` and read the same way at the call site (``settings.QDRANT_URL``).
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from the process environment, falling back to ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # .env also carries variables for later phases (verifier, API auth,
        # database). Without this, BaseSettings' default extra="forbid" would
        # reject them before those tasks add the fields.
        extra="ignore",
    )

    # --- LLM provider ---
    ANTHROPIC_API_KEY: SecretStr = Field(
        description="Anthropic API key. Required; fail fast rather than 401 at request time.",
    )
    LLM_MODEL: str = Field(
        default="claude-sonnet-5",
        description="Planner model. Exact model ID, no date suffix.",
    )

    # --- Vector store ---
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant REST endpoint.",
    )
    COLLECTION_NAME: str = Field(
        default="sec_filings",
        description="Qdrant collection holding 10-K chunks.",
    )

    # --- Embeddings ---
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed model. Truncates at 512 tokens, so chunk size must match.",
    )

    # --- Agent state ---
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis URL for the LangGraph checkpointer. Needs JSON and Search, which are in Redis 8 core.",
    )
    MAX_AGENT_LOOPS: int = Field(
        default=6,
        ge=1,
        description="Hard ceiling on planner/tool iterations per query.",
    )

    # --- SEC filings ---
    SEC_IDENTITY: str = Field(
        description="Contact identity the SEC requires on every EDGAR request, e.g. 'Name email'.",
    )

    # --- Tracing ---
    LANGSMITH_TRACING: bool = Field(
        default=False,
        description="Enable LangSmith tracing.",
    )
    LANGSMITH_API_KEY: SecretStr | None = Field(
        default=None,
        description="LangSmith API key. Optional; unset disables tracing.",
    )
    LANGSMITH_PROJECT: str = Field(
        default="finai-agent-dev",
        description="LangSmith project name.",
    )
