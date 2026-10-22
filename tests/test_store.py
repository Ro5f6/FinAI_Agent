"""Tests for the Qdrant chunk collection.

Offline tests use qdrant-client's in-memory mode. Tests marked ``qdrant`` use
the server from docker compose: ``uv run pytest -m qdrant``.
"""

import typing

import pytest
import voyageai
from qdrant_client import QdrantClient, models

from finai.config import settings
from finai.ingestion import embed
from finai.ingestion.chunk import ChunkPayload, chunk_filing
from finai.ingestion.fetch import FilingMeta
from finai.ingestion.store import (
    DISTANCE,
    PAYLOAD_INDEXES,
    CollectionMismatchError,
    IngestCountError,
    ensure_collection,
    ensure_payload_indexes,
    indexed_years,
    qdrant_client,
    search,
    upsert_filing,
)


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_creates_the_collection_from_settings(client: QdrantClient) -> None:
    assert ensure_collection(client) is True

    vectors = client.get_collection(settings.COLLECTION_NAME).config.params.vectors
    assert isinstance(vectors, models.VectorParams)
    assert vectors.size == settings.EMBEDDING_DIM == 1024
    assert vectors.distance == DISTANCE == models.Distance.COSINE


def test_existing_matching_collection_is_left_alone(client: QdrantClient) -> None:
    ensure_collection(client, name="c", dim=8)
    client.upsert("c", points=[models.PointStruct(id=1, vector=[0.1] * 8)])

    assert ensure_collection(client, name="c", dim=8) is False
    assert client.count("c").count == 1


def test_wrong_vector_size_raises_instead_of_recreating(client: QdrantClient) -> None:
    """A 384-dim collection from bge-small must not be wiped by voyage-4's 1024."""
    client.create_collection(
        "c", vectors_config=models.VectorParams(size=384, distance=DISTANCE)
    )
    client.upsert("c", points=[models.PointStruct(id=1, vector=[0.1] * 384)])

    with pytest.raises(CollectionMismatchError, match="384"):
        ensure_collection(client, name="c", dim=1024)
    assert client.count("c").count == 1


def test_wrong_distance_raises(client: QdrantClient) -> None:
    client.create_collection(
        "c", vectors_config=models.VectorParams(size=8, distance=models.Distance.DOT)
    )

    with pytest.raises(CollectionMismatchError, match="Dot"):
        ensure_collection(client, name="c", dim=8)


def test_named_vectors_raise(client: QdrantClient) -> None:
    client.create_collection(
        "c",
        vectors_config={"text": models.VectorParams(size=8, distance=DISTANCE)},
    )

    with pytest.raises(CollectionMismatchError, match="named vectors"):
        ensure_collection(client, name="c", dim=8)


class _RecordingClient:
    """Wraps an in-memory client, recording index requests it would ignore."""

    def __init__(self, inner: QdrantClient) -> None:
        self._inner = inner
        self.requested: list[tuple[str, models.PayloadSchemaType]] = []

    def get_collection(self, name: str) -> models.CollectionInfo:
        return self._inner.get_collection(name)

    def create_payload_index(self, collection_name: str, field_name: str, field_schema):
        self.requested.append((field_name, field_schema))


def test_requests_the_planned_indexes(client: QdrantClient) -> None:
    ensure_collection(client, name="c", dim=8)
    recorder = _RecordingClient(client)

    created = ensure_payload_indexes(recorder, name="c")

    assert recorder.requested == [
        ("ticker", models.PayloadSchemaType.KEYWORD),
        ("fiscal_year", models.PayloadSchemaType.INTEGER),
        ("section", models.PayloadSchemaType.KEYWORD),
        ("ciks", models.PayloadSchemaType.INTEGER),
    ]
    assert created == ["ticker", "fiscal_year", "section", "ciks"]


def test_indexed_fields_are_payload_fields_of_the_right_type() -> None:
    """An index on a field the payload doesn't carry would filter nothing."""
    fields = ChunkPayload.model_fields
    python_type = {
        models.PayloadSchemaType.KEYWORD: str,
        models.PayloadSchemaType.INTEGER: int,
    }
    for field, schema in PAYLOAD_INDEXES.items():
        assert field in fields
        annotation = fields[field].annotation
        # A list field is indexed by its element type (ciks: list[int]).
        element = (
            typing.get_args(annotation)[0]
            if typing.get_origin(annotation) is list
            else annotation
        )
        assert element is python_type[schema]


def test_qdrant_client_is_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new client costs ~25 ms (a version check); a reused one ~1 ms.

    A stand-in replaces QdrantClient: a real one starts that version check
    against the server, a network call this offline test must not make.
    """
    from finai.ingestion import store

    built = []
    monkeypatch.setattr(
        store, "QdrantClient", lambda url: built.append(url) or object()
    )
    qdrant_client.cache_clear()

    assert qdrant_client() is qdrant_client()
    assert built == [settings.QDRANT_URL]
    qdrant_client.cache_clear()


def _meta(accession: str) -> FilingMeta:
    return FilingMeta(
        "AAPL", accession, "10-K", "2025-10-31", "2025-09-27", 2025, (320193,)
    )


def _filing(accession: str, n_paragraphs: int) -> list[ChunkPayload]:
    body = "\n\n".join(f"Paragraph {i} about supply risk." for i in range(n_paragraphs))
    # 8-token chunks: one paragraph per chunk.
    return chunk_filing(_meta(accession), {"Item 1A": body}, 8, 0)


def _vectors(payloads: list[ChunkPayload]) -> list[list[float]]:
    return [[1.0, float(p.chunk_index), 0.0, 0.0] for p in payloads]


@pytest.fixture
def small(client: QdrantClient) -> QdrantClient:
    ensure_collection(client, name="c", dim=4)
    return client


def test_upsert_writes_points_with_their_payload(small: QdrantClient) -> None:
    payloads = _filing("A-1", 3)

    assert upsert_filing(small, payloads, _vectors(payloads), name="c") == 0

    assert small.count("c").count == 3
    [point] = small.retrieve("c", ids=[payloads[1].point_id()])
    assert point.payload == payloads[1].model_dump()


def test_reingesting_overwrites_instead_of_duplicating(small: QdrantClient) -> None:
    payloads = _filing("A-1", 3)
    upsert_filing(small, payloads, _vectors(payloads), name="c")

    upsert_filing(small, payloads, _vectors(payloads), name="c")

    assert small.count("c").count == 3


def test_fewer_chunks_on_reingest_removes_the_extras(small: QdrantClient) -> None:
    upsert_filing(small, _filing("A-1", 5), _vectors(_filing("A-1", 5)), name="c")
    upsert_filing(small, _filing("B-2", 2), _vectors(_filing("B-2", 2)), name="c")

    shorter = _filing("A-1", 3)
    removed = upsert_filing(small, shorter, _vectors(shorter), name="c")

    assert removed == 2
    by_filing = models.Filter(
        must=[
            models.FieldCondition(key="accession", match=models.MatchValue(value="A-1"))
        ]
    )
    assert small.count("c", count_filter=by_filing).count == 3
    assert small.count("c").count == 5  # the other filing's 2 points are untouched


AAPL, NVDA = 320193, 1045810


def _seed_two_companies(client: QdrantClient) -> None:
    """AAPL FY2025 and FY2024 plus NVDA FY2025, with chunk i's vector pointing
    at angle i so the nearest chunk is predictable."""
    for ticker, accession, year, cik in (
        ("AAPL", "A-25", 2025, AAPL),
        ("AAPL", "A-24", 2024, AAPL),
        ("NVDA", "N-25", 2025, NVDA),
    ):
        meta = FilingMeta(
            ticker, accession, "10-K", f"{year}-10-31", f"{year}-09-27", year, (cik,)
        )
        sections = {"Item 1A": "Risk one.\n\nRisk two.", "Item 7": "Results one."}
        payloads = chunk_filing(meta, sections, 8, 0)
        upsert_filing(client, payloads, _vectors(payloads), name="c")


def test_search_is_limited_to_the_company(small: QdrantClient) -> None:
    _seed_two_companies(small)

    hits = search(small, [1.0, 0.0, 0.0, 0.0], AAPL, k=10, name="c")

    assert hits
    assert {h.chunk.ticker for h in hits} == {"AAPL"}
    assert all(isinstance(h.chunk, ChunkPayload) for h in hits)


def test_search_filters_section_and_year(small: QdrantClient) -> None:
    _seed_two_companies(small)

    hits = search(
        small,
        [1.0, 0.0, 0.0, 0.0],
        AAPL,
        section="Item 1A",
        fiscal_year=2024,
        k=10,
        name="c",
    )

    assert [(h.chunk.accession, h.chunk.section) for h in hits] == [("A-24", "Item 1A")]


def test_search_returns_best_first_and_respects_k(small: QdrantClient) -> None:
    _seed_two_companies(small)

    hits = search(small, [1.0, 1.0, 0.0, 0.0], AAPL, k=3, name="c")

    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


@pytest.mark.qdrant
@pytest.mark.voyage
def test_supply_chain_question_finds_apples_risk_factors() -> None:
    """Task 1.2.17. Needs AAPL ingested into the compose Qdrant."""
    from finai.ingestion.embed import embed_query

    vector = embed_query("What are Apple's main supply chain risks?")
    hits = search(qdrant_client(), vector, 320193)  # Apple's CIK

    assert len(hits) == 5
    assert hits[0].chunk.section == "Item 1A"
    assert any("single or limited sources" in h.chunk.text for h in hits)


def test_indexed_years_are_newest_first(small: QdrantClient) -> None:
    """Facets come back by count; the years must come back by year."""
    _seed_two_companies(small)

    assert indexed_years(small, AAPL, name="c") == [2025, 2024]
    assert indexed_years(small, NVDA, name="c") == [2025]
    assert indexed_years(small, 1318605, name="c") == []  # TSLA: not stored


def _seed_ranked(client: QdrantClient, n: int = 40) -> list[ChunkPayload]:
    """``n`` AAPL chunks whose embedding rank is their chunk_index."""
    body = "\n\n".join(f"Paragraph {i} about supply risk." for i in range(n))
    payloads = chunk_filing(_meta("A-1"), {"Item 1A": body}, 8, 0)
    vectors = [[1.0, i / 1000, 0.0, 0.0] for i in range(len(payloads))]
    upsert_filing(client, payloads, vectors, name="c")
    return payloads


def test_rerank_reorders_the_candidates(small: QdrantClient, monkeypatch) -> None:
    _seed_ranked(small)
    seen = []

    def reverse(question, texts):
        seen.append(len(texts))
        return [(i, float(i)) for i in reversed(range(len(texts)))]

    monkeypatch.setattr(embed, "rerank", reverse)

    hits = search(small, [1.0, 0.0, 0.0, 0.0], AAPL, k=3, name="c", question="supply")

    assert seen == [settings.RERANK_CANDIDATES]
    # The reranker's top pick was the embedding's 30th.
    assert [h.chunk.chunk_index for h in hits] == [29, 28, 27]
    assert [h.relevance for h in hits] == [29.0, 28.0, 27.0]


def test_without_a_question_there_is_no_rerank(
    small: QdrantClient, monkeypatch
) -> None:
    _seed_ranked(small)
    monkeypatch.setattr(embed, "rerank", lambda q, t: pytest.fail("reranked"))

    hits = search(small, [1.0, 0.0, 0.0, 0.0], AAPL, k=3, name="c")

    assert [h.chunk.chunk_index for h in hits] == [0, 1, 2]
    assert all(h.relevance is None for h in hits)


@pytest.mark.parametrize(
    "error",
    [
        voyageai.error.RateLimitError("slow down"),
        voyageai.error.ServiceUnavailableError("down"),
        voyageai.error.Timeout("too slow"),
        voyageai.error.APIConnectionError("no route"),
    ],
)
def test_transient_rerank_failure_falls_back_to_embedding_order(
    small: QdrantClient, monkeypatch, error
) -> None:
    _seed_ranked(small)

    def fail(question, texts):
        raise error

    monkeypatch.setattr(embed, "rerank", fail)

    hits = search(small, [1.0, 0.0, 0.0, 0.0], AAPL, k=3, name="c", question="supply")

    assert [h.chunk.chunk_index for h in hits] == [0, 1, 2]
    assert all(h.relevance is None for h in hits)


@pytest.mark.parametrize(
    "error",
    [
        voyageai.error.AuthenticationError("bad key"),
        voyageai.error.InvalidRequestError("unknown model rerank-9"),
    ],
)
def test_configuration_errors_are_not_hidden(
    small: QdrantClient, monkeypatch, error
) -> None:
    """A wrong key or model must not silently turn reranking off."""
    _seed_ranked(small)

    def fail(question, texts):
        raise error

    monkeypatch.setattr(embed, "rerank", fail)

    with pytest.raises(type(error)):
        search(small, [1.0, 0.0, 0.0, 0.0], AAPL, k=3, name="c", question="supply")


def test_rerank_client_fails_fast() -> None:
    """Search waits on it; the ingest client's 5 retries could stall a minute."""
    embed._rerank_client.cache_clear()

    client = embed._rerank_client()

    assert client.max_retries == embed.RERANK_RETRIES == 1
    assert embed.RERANK_TIMEOUT_S <= 5
    embed._rerank_client.cache_clear()


class _LosingClient:
    """Passes calls through but drops the last point of every upsert, as a
    failed or partial write would."""

    def __init__(self, inner: QdrantClient) -> None:
        self._inner = inner

    def upsert(self, collection_name, points, wait=True):
        return self._inner.upsert(collection_name, points=points[:-1], wait=wait)

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


def test_count_mismatch_raises_and_names_the_filing(small: QdrantClient) -> None:
    payloads = _filing("A-1", 3)

    with pytest.raises(IngestCountError, match="AAPL A-1: 2 points stored, 3 chunks"):
        upsert_filing(_LosingClient(small), payloads, _vectors(payloads), name="c")


def test_count_is_checked_per_filing(small: QdrantClient) -> None:
    """Another filing's points must not mask a short write."""
    other = _filing("B-2", 4)
    upsert_filing(small, other, _vectors(other), name="c")
    payloads = _filing("A-1", 3)

    with pytest.raises(IngestCountError, match="A-1"):
        upsert_filing(_LosingClient(small), payloads, _vectors(payloads), name="c")


def test_mismatched_lengths_raise(small: QdrantClient) -> None:
    payloads = _filing("A-1", 3)

    with pytest.raises(ValueError, match="3 payloads but 2 vectors"):
        upsert_filing(small, payloads, _vectors(payloads)[:2], name="c")


def test_one_call_writes_one_filing(small: QdrantClient) -> None:
    mixed = _filing("A-1", 2) + _filing("B-2", 2)

    with pytest.raises(ValueError, match="one filing"):
        upsert_filing(small, mixed, _vectors(mixed), name="c")


@pytest.fixture
def server_collection() -> tuple[QdrantClient, str]:
    """A throwaway collection on the compose Qdrant, deleted afterwards."""
    client = qdrant_client()
    name = "test_store_indexes"
    if client.collection_exists(name):
        client.delete_collection(name)
    ensure_collection(client, name=name, dim=4)
    yield client, name
    client.delete_collection(name)


@pytest.mark.qdrant
def test_indexes_exist_on_the_server(server_collection) -> None:
    client, name = server_collection

    assert ensure_payload_indexes(client, name=name) == list(PAYLOAD_INDEXES)
    assert ensure_payload_indexes(client, name=name) == []

    schema = client.get_collection(name).payload_schema
    assert {f: s.data_type for f, s in schema.items()} == PAYLOAD_INDEXES


@pytest.mark.qdrant
def test_wrong_index_type_is_corrected(server_collection) -> None:
    client, name = server_collection
    client.create_payload_index(
        name, "fiscal_year", field_schema=models.PayloadSchemaType.KEYWORD
    )

    assert "fiscal_year" in ensure_payload_indexes(client, name=name)
    schema = client.get_collection(name).payload_schema
    assert schema["fiscal_year"].data_type == models.PayloadSchemaType.INTEGER


@pytest.mark.qdrant
def test_ticker_and_year_range_filter(server_collection) -> None:
    client, name = server_collection
    ensure_payload_indexes(client, name=name)
    points = [
        models.PointStruct(
            id=i,
            vector=[0.1, 0.2, 0.3, float(i)],
            payload={"ticker": t, "fiscal_year": y, "section": "Item 1A"},
        )
        for i, (t, y) in enumerate(
            [("AAPL", 2023), ("AAPL", 2024), ("AAPL", 2025), ("NVDA", 2025)]
        )
    ]
    client.upsert(name, points=points)

    since_2024 = models.Filter(
        must=[
            models.FieldCondition(key="ticker", match=models.MatchValue(value="AAPL")),
            models.FieldCondition(key="fiscal_year", range=models.Range(gte=2024)),
        ]
    )

    assert client.count(name, count_filter=since_2024).count == 2
    schema = client.get_collection(name).payload_schema
    assert schema["ticker"].points == 4
    assert schema["fiscal_year"].points == 4


@pytest.mark.qdrant
def test_collection_on_the_compose_qdrant() -> None:
    client = qdrant_client()

    ensure_collection(client)

    assert ensure_collection(client) is False
    info = client.get_collection(settings.COLLECTION_NAME)
    assert info.config.params.vectors.size == settings.EMBEDDING_DIM
    ensure_payload_indexes(client)
    schema = client.get_collection(settings.COLLECTION_NAME).payload_schema
    assert {f: s.data_type for f, s in schema.items()} == PAYLOAD_INDEXES


def test_indexed_years_of_a_missing_collection_is_empty() -> None:
    """A fresh deployment before the first ingest: no collection yet. The
    first question must queue the company, not fail (found by 1.3.30)."""
    assert indexed_years(QdrantClient(":memory:"), AAPL, name="never_created") == []


def test_share_classes_and_successors_find_one_filing(small: QdrantClient) -> None:
    """D-043: one filing, found by every CIK it carries, stored once."""
    meta = FilingMeta(
        "XOM", "X-25", "10-K", "2026-02-18", "2025-12-31", 2025, (34088, 2115436)
    )
    payloads = chunk_filing(meta, {"Item 1A": "Oil price risk."}, 8, 0)
    upsert_filing(small, payloads, _vectors(payloads), name="c")

    for cik in (34088, 2115436):  # the filer, and the company XOM maps to now
        assert indexed_years(small, cik, name="c") == [2025]
        assert search(small, [1.0, 0.0, 0.0, 0.0], cik, name="c")
