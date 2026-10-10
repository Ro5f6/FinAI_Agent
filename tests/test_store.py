"""Tests for the Qdrant chunk collection.

Offline tests use qdrant-client's in-memory mode. Tests marked ``qdrant`` use
the server from docker compose: ``uv run pytest -m qdrant``.
"""

import pytest
from qdrant_client import QdrantClient, models

from finai.config import settings
from finai.ingestion.store import (
    DISTANCE,
    CollectionMismatchError,
    ensure_collection,
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


@pytest.mark.qdrant
def test_collection_on_the_compose_qdrant() -> None:
    client = qdrant_client()

    ensure_collection(client)

    assert ensure_collection(client) is False
    info = client.get_collection(settings.COLLECTION_NAME)
    assert info.config.params.vectors.size == settings.EMBEDDING_DIM
