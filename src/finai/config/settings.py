"""Application configuration.

Every configuration value the app reads is declared here. Nothing else calls
``os.getenv``; import the ``settings`` singleton instead.

Field names are uppercase so they map one-to-one onto the variable names in
``.env.example`` and read the same way at the call site (``settings.QDRANT_URL``).
"""

from pathlib import Path
from typing import Literal

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
        default="claude-sonnet-5-5",
        description=(
            "Planner model. Exact model ID, no date suffix. claude-haiku-5-5 is 20x "
            "cheaper for development; evaluate with the model you deploy."
        ),
    )
    LLM_EFFORT: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="medium",
        description=(
            "Planner reasoning effort. Set explicitly: defaults differ by model "
            "(Sonnet 5.5 high, Opus 5.5 and Haiku 5.5 medium)."
        ),
    )
    LLM_MAX_TOKENS: int = Field(
        default=16_000,
        ge=1,
        description="Output cap per planner response; ~16K keeps non-streaming calls under SDK timeouts.",
    )
    LLM_REFUSAL_FALLBACK: bool = Field(
        default=True,
        description="Re-run a safety-declined request on a fallback model (server-side, Claude API only).",
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
    RERANK_MODEL: str = Field(
        default="rerank-2.5",
        description="Voyage reranker applied to the top RERANK_CANDIDATES of each 10-K search.",
    )
    RERANK_CANDIDATES: int = Field(
        default=30,
        ge=1,
        le=1000,
        description="Passages the reranker scores per search. 50 and 100 added latency, not hits.",
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

    # --- Market data ---
    FINNHUB_API_KEY: SecretStr = Field(
        description="Finnhub API key for live quotes. Free plan: US listings, 60 calls/min.",
    )
    ALPACA_API_KEY: SecretStr = Field(
        description="Alpaca key ID for daily price history. A free paper account's key works.",
    )
    ALPACA_SECRET_KEY: SecretStr = Field(
        description="Alpaca secret key, paired with ALPACA_API_KEY.",
    )

    # --- Agent state ---
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis URL for the LangGraph checkpointer. Needs JSON and Search, which are in Redis 8 core.",
    )
    CHECKPOINT_TTL_MINUTES: int = Field(
        default=30 * 24 * 60,
        ge=1,
        description=(
            "Saved conversations expire this many MINUTES after last use (reading "
            "a thread renews it). The Redis saver's ttl is in minutes, not seconds."
        ),
    )
    INGEST_STATUS_TTL_MINUTES: int = Field(
        default=24 * 60,
        ge=1,
        description=(
            "How long a ticker's on-demand ingestion status (queued, ingesting, "
            "failed) is kept. Until it expires the ticker isn't queued again, so "
            "a failed fetch is retried at most once per this period."
        ),
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
