"""The Qdrant collection that holds 10-K chunks.

Verified against qdrant-client 1.19.1 and server v1.19.1 (docs/CONTEXT.md).
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import NamedTuple

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from finai.config import settings
from finai.ingestion import embed
from finai.ingestion.chunk import ChunkPayload

log = logging.getLogger(__name__)

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


class IngestCountError(RuntimeError):
    """A filing's point count differs from the number of chunks written."""


class SearchHit(NamedTuple):
    """One retrieved chunk, its cosine similarity to the query and, when the
    search was reranked, the reranker's relevance score (which set the order)."""

    score: float
    chunk: ChunkPayload
    relevance: float | None = None


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

    Afterwards the filing's points are counted exactly and must equal the
    chunks written (1.2.16). The check is per filing rather than for the
    whole collection, so it holds with many filings stored and names the one
    that is wrong.
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

    accession = accessions.pop()
    this_filing = models.FieldCondition(
        key="accession", match=models.MatchValue(value=accession)
    )
    stale = models.Filter(
        must=[this_filing],
        must_not=[models.HasIdCondition(has_id=[p.id for p in points])],
    )
    removed = client.count(name, count_filter=stale, exact=True).count
    if removed:
        client.delete(
            name, points_selector=models.FilterSelector(filter=stale), wait=True
        )

    stored = client.count(
        name, count_filter=models.Filter(must=[this_filing]), exact=True
    ).count
    if stored != len(points):
        raise IngestCountError(
            f"{payloads[0].ticker} {accession}: {stored} points stored, "
            f"{len(points)} chunks written"
        )
    return removed


def search(
    client: QdrantClient,
    vector: Sequence[float],
    ticker: str,
    section: str | None = None,
    fiscal_year: int | None = None,
    k: int = 5,
    name: str | None = None,
    question: str | None = None,
) -> list[SearchHit]:
    """Return the ``k`` chunks most relevant to the query for one company.

    With ``question``, the top ``RERANK_CANDIDATES`` by embedding are reranked
    and the best ``k`` returned: on the 18-question eval this took top-5 hits
    from 14 to 17 and top-1 from 9 to 14 (docs/evals/retrieval-reranking.md).
    If the reranker is briefly unavailable, the embedding order is returned.

    ``ticker`` is required: every 10-K search is about one company, and an
    unfiltered search would rank other companies' risk factors alongside its
    own. ``section`` and ``fiscal_year`` narrow it further, using the payload
    indexes from ``ensure_payload_indexes``. ``QdrantClient.search`` no longer
    exists in qdrant-client 1.19; this uses ``query_points``.
    """
    must = [
        models.FieldCondition(
            key="ticker", match=models.MatchValue(value=ticker.upper())
        )
    ]
    if section is not None:
        must.append(
            models.FieldCondition(key="section", match=models.MatchValue(value=section))
        )
    if fiscal_year is not None:
        must.append(
            models.FieldCondition(
                key="fiscal_year", match=models.MatchValue(value=fiscal_year)
            )
        )
    response = client.query_points(
        collection_name=name or settings.COLLECTION_NAME,
        query=list(vector),
        query_filter=models.Filter(must=must),
        limit=max(k, settings.RERANK_CANDIDATES) if question else k,
        with_payload=True,
    )
    hits = [
        SearchHit(point.score, ChunkPayload(**point.payload))
        for point in response.points
    ]
    if not question or len(hits) < 2:
        return hits[:k]
    try:
        order = embed.rerank(question, [h.chunk.text for h in hits])
    except embed.TRANSIENT_ERRORS as error:
        log.warning("reranker unavailable, using embedding order: %s", error)
        return hits[:k]
    return [hits[i]._replace(relevance=score) for i, score in order[:k]]


def indexed_years(
    client: QdrantClient, ticker: str, name: str | None = None
) -> list[int]:
    """Fiscal years with chunks for ``ticker``, newest first; [] if none.

    Uses Qdrant's facet count over the integer index, so nothing is scanned.
    Facets come back ordered by count, not value, hence the sort.

    A collection that doesn't exist yet, as on a fresh deployment before the
    first ingest, has no years: the server answers 404 and the in-memory
    client raises ValueError, so either is checked against
    ``collection_exists`` (1.5 ms) before being treated as empty.
    """
    name = name or settings.COLLECTION_NAME
    try:
        response = _facet_years(client, ticker, name)
    except (UnexpectedResponse, ValueError):
        if not client.collection_exists(name):
            return []
        raise
    return sorted((int(hit.value) for hit in response.hits), reverse=True)


def _facet_years(client: QdrantClient, ticker: str, name: str) -> models.FacetResponse:
    return client.facet(
        collection_name=name,
        key="fiscal_year",
        facet_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="ticker", match=models.MatchValue(value=ticker.upper())
                )
            ]
        ),
        limit=100,
        exact=True,
    )
