"""Application configuration.

Every configuration value the app reads is declared here. Nothing else calls
``os.getenv``; import the ``settings`` singleton instead.

Field names are uppercase so they map one-to-one onto the variable names in
``.env.example`` and read the same way at the call site (``settings.QDRANT_URL``).
"""

from pathlib import Path

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
    # voyage-4 won the 18-question evaluation (docs/evals/embedding-models.md)
    # and needs ~100 MB in the API pod against ~900 MB for the best local model.
    VOYAGE_API_KEY: SecretStr = Field(
        description="Voyage AI API key. Required: every 10-K search embeds its query through it.",
    )
    EMBEDDING_MODEL: str = Field(
        default="voyage-4",
        description="Voyage embedding model. Changing it means re-embedding the collection.",
    )
    EMBEDDING_MAX_TOKENS: int = Field(
        default=32_000,
        ge=1,
        description="Model context length in its own tokens; longer input is truncated by the API.",
    )
    EMBEDDING_DIM: int = Field(
        default=1024,
        ge=1,
        description="Vector size; sets the Qdrant collection's size. voyage-4 also offers 256, 512, 2048.",
    )
    CHUNK_TOKENS: int = Field(
        default=1_000,
        ge=1,
        description="Target chunk size. Long-context model, so ~1,000 tokens rather than 400-500.",
    )
    CHUNK_OVERLAP_TOKENS: int = Field(
        default=100,
        ge=0,
        description="Tokens shared between consecutive chunks, about 10% of CHUNK_TOKENS.",
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
    RAW_FILINGS_DIR: Path = Field(
        default=Path("data/raw"),
        description="Archive of fetched filings, gzipped, for provenance and manual inspection.",
    )
    EDGAR_LOCAL_DATA_DIR: Path = Field(
        default=Path("data/edgar"),
        description="edgartools local store. This is the cache that makes re-runs free.",
    )
    EDGAR_RATE_LIMIT_PER_SEC: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "SEC caps requests at 10/s. edgartools defaults to 9; 5 leaves headroom. "
            "The limiter is per process, so parallel workers multiply this."
        ),
    )
    EDGAR_ALLOW_NETWORK_FALLBACK: bool = Field(
        default=True,
        description="False forbids all SEC requests, serving only from the local store.",
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
