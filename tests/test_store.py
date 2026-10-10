"""Tests for the Qdrant chunk collection.

Offline tests use qdrant-client's in-memory mode. Tests marked ``qdrant`` use
the server from docker compose: ``uv run pytest -m qdrant``.
"""

import pytest
from qdrant_client import QdrantClient, models

from finai.config import settings
from finai.ingestion.chunk import ChunkPayload
from finai.ingestion.store import (
    DISTANCE,
    PAYLOAD_INDEXES,
    CollectionMismatchError,
    ensure_collection,
    ensure_payload_indexes,
    qdrant_client,
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
    ]
    assert created == ["ticker", "fiscal_year", "section"]


def test_indexed_fields_are_payload_fields_of_the_right_type() -> None:
    """An index on a field the payload doesn't carry would filter nothing."""
    fields = ChunkPayload.model_fields
    python_type = {
        models.PayloadSchemaType.KEYWORD: str,
        models.PayloadSchemaType.INTEGER: int,
    }
    for field, schema in PAYLOAD_INDEXES.items():
        assert field in fields
        assert fields[field].annotation is python_type[schema]


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
