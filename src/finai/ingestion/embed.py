"""Embedding chunks and queries with Voyage AI.

Verified against voyageai 0.5.0 (docs/CONTEXT.md). Documents and queries are
embedded with ``input_type`` "document" and "query": Voyage prepends its own
retrieval prompt for each, so no prefixes are added here.
"""

from collections.abc import Iterator
from functools import lru_cache

import voyageai

from finai.config import settings

# Per request, voyage-4 accepts at most 1,000 texts and 320K tokens. Batches
# stay well under the token cap: a 1,000-token chunk is ~1,086 Voyage tokens.
MAX_BATCH_TEXTS = 1_000
MAX_BATCH_TOKENS = 100_000
# The client retries rate limits, 503s and timeouts with exponential backoff
# (1-16 s) when this is above 0; its default is 0, meaning no retries.
MAX_RETRIES = 5


@lru_cache(maxsize=1)
def voyage_client() -> voyageai.Client:
    """One client per process.

    The client keeps a ``requests`` session per thread, so reusing it reuses
    connections.
    """
    return voyageai.Client(
        api_key=settings.VOYAGE_API_KEY.get_secret_value(),
        max_retries=MAX_RETRIES,
    )


def batches(
    texts: list[str],
    token_counts: list[int],
    max_tokens: int = MAX_BATCH_TOKENS,
    max_texts: int = MAX_BATCH_TEXTS,
) -> Iterator[list[str]]:
    """Group texts into requests under Voyage's per-request limits."""
    batch: list[str] = []
    used = 0
    for text, tokens in zip(texts, token_counts, strict=True):
        if batch and (used + tokens > max_tokens or len(batch) == max_texts):
            yield batch
            batch, used = [], 0
        batch.append(text)
        used += tokens
    if batch:
        yield batch


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts, in the order given."""
    client = voyage_client()
    # count_tokens runs locally with Voyage's tokenizer; one call per text gives
    # the per-text counts that batching needs.
    counts = [client.count_tokens([t], model=settings.EMBEDDING_MODEL) for t in texts]
    vectors: list[list[float]] = []
    for batch in batches(texts, counts):
        result = client.embed(
            batch,
            model=settings.EMBEDDING_MODEL,
            input_type="document",
            output_dimension=settings.EMBEDDING_DIM,
        )
        vectors.extend(result.embeddings)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed one search question."""
    result = voyage_client().embed(
        [text],
        model=settings.EMBEDDING_MODEL,
        input_type="query",
        output_dimension=settings.EMBEDDING_DIM,
    )
    return result.embeddings[0]
