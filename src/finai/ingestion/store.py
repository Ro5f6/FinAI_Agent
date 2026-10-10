"""The Qdrant collection that holds 10-K chunks.

Verified against qdrant-client 1.19.1 and server v1.19.1 (docs/CONTEXT.md).
"""

from collections.abc import Sequence
from functools import lru_cache

from qdrant_client import QdrantClient, models

from finai.config import settings
from finai.ingestion.chunk import ChunkPayload

DISTANCE = models.Distance.COSINE

# Fields that searches filter on. fiscal_year is an integer index so year
# filters and ranges ("since 2023") use it; ChunkPayload rejects a string year.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "ticker": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
    "section": models.PayloadSchemaType.KEYWORD,
}


class CollectionMismatchError(RuntimeError):
    """The collection exists with a different vector size or distance."""


# Points per upsert request. Each point carries a 1024-float vector and up to
# ~4 KB of text, so 256 keeps a request around 5 MB.
UPSERT_BATCH = 256


@lru_cache(maxsize=1)
def qdrant_client() -> QdrantClient:
    """One client per process for the Qdrant at ``settings.QDRANT_URL``.

    Creating a client checks the server version: 24.6 ms per call against
    0.9 ms for a reused client, measured on the compose Qdrant. The client's
    httpx pool keeps connections open, except to localhost, where the client
    turns keep-alive off itself.
    """
    return QdrantClient(url=settings.QDRANT_URL)


def ensure_collection(
    client: QdrantClient,
    name: str | None = None,
    dim: int | None = None,
) -> bool:
    """Create the chunk collection if it is missing; return whether it was created.

    An existing collection with another vector size or distance raises rather
    than being recreated. That means the embedding model changed (bge-small
    gave 384 dims, voyage-4 gives 1024), and recreating would silently delete
    every stored point; re-embedding is a deliberate step.
    """
    name = name or settings.COLLECTION_NAME
    dim = dim or settings.EMBEDDING_DIM

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=DISTANCE),
        )
        return True

    vectors = client.get_collection(name).config.params.vectors
    if not isinstance(vectors, models.VectorParams):
        raise CollectionMismatchError(
            f"collection {name!r} uses named vectors; expected one unnamed vector"
        )
    if vectors.size != dim or vectors.distance != DISTANCE:
        raise CollectionMismatchError(
            f"collection {name!r} has {vectors.size} dims, {vectors.distance}; "
            f"settings expect {dim} dims, {DISTANCE}. Re-embed into a new "
            "collection or delete this one deliberately."
        )
    return False


def ensure_payload_indexes(client: QdrantClient, name: str | None = None) -> list[str]:
    """Index the filter fields; return the fields whose index was created.

    An index that exists with the wrong type is recreated with the right one:
    Qdrant replaces it in place without an error, and only the index changes,
    not the stored points. Local (in-memory) Qdrant ignores payload indexes,
    so their effect is only testable against a server.
    """
    name = name or settings.COLLECTION_NAME
    existing = client.get_collection(name).payload_schema
    created = []
    for field, schema in PAYLOAD_INDEXES.items():
        current = existing.get(field)
        if current is not None and current.data_type == schema:
            continue
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema
        )
        created.append(field)
    return created


def upsert_filing(
    client: QdrantClient,
    payloads: Sequence[ChunkPayload],
    vectors: Sequence[Sequence[float]],
    name: str | None = None,
) -> int:
    """Write one filing's chunks; return how many stale points were removed.

    Points are upserted first and leftovers deleted after, so a search during
    a re-ingest never finds the filing empty. Leftovers are the filing's
    points whose IDs the new chunking didn't produce, e.g. after a chunker
    change that yields fewer chunks.
    """
    if len(payloads) != len(vectors):
        raise ValueError(f"{len(payloads)} payloads but {len(vectors)} vectors")
    if not payloads:
        return 0
    accessions = {p.accession for p in payloads}
    if len(accessions) != 1:
        raise ValueError(f"expected one filing, got {sorted(accessions)}")
    name = name or settings.COLLECTION_NAME

    points = [
        models.PointStruct(id=p.point_id(), vector=list(v), payload=p.model_dump())
        for p, v in zip(payloads, vectors, strict=True)
    ]
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(name, points=points[start : start + UPSERT_BATCH], wait=True)

    stale = models.Filter(
        must=[
            models.FieldCondition(
                key="accession", match=models.MatchValue(value=accessions.pop())
            )
        ],
        must_not=[models.HasIdCondition(has_id=[p.id for p in points])],
    )
    removed = client.count(name, count_filter=stale, exact=True).count
    if removed:
        client.delete(name, points_selector=models.FilterSelector(filter=stale))
    return removed
