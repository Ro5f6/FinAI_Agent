"""The Qdrant collection that holds 10-K chunks.

Verified against qdrant-client 1.19.1 and server v1.19.1 (docs/CONTEXT.md).
"""

from qdrant_client import QdrantClient, models

from finai.config import settings

DISTANCE = models.Distance.COSINE


class CollectionMismatchError(RuntimeError):
    """The collection exists with a different vector size or distance."""


def qdrant_client() -> QdrantClient:
    """Client for the Qdrant at ``settings.QDRANT_URL``."""
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
