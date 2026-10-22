"""Tests for snapshotting and restoring the 10-K collection.

Offline tests cover what needs no server: the manifest and the integrity
check that runs before any upload. ``-m qdrant`` runs a full round trip on
the compose Qdrant with a small throwaway collection: create, download,
restore under another name, compare, and a damaged file refused.
"""

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from finai.config import settings
from finai.ingestion import snapshot
from finai.ingestion.snapshot import Manifest, SnapshotError, read_manifest


def _write(directory: Path, content: bytes, sha256: str) -> Path:
    path = directory / "c-1.snapshot"
    path.write_bytes(content)
    manifest = Manifest(
        collection="c",
        snapshot=path.name,
        sha256=sha256,
        bytes=len(content),
        points=3,
        vector_size=4,
        distance="Cosine",
        embedding_model="voyage-4",
        created_at="2026-10-22T17:08:32+00:00",
    )
    path.with_suffix(".json").write_text(json.dumps(manifest._asdict()))
    return path


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = _write(tmp_path, b"data", "x")
    manifest = read_manifest(path)
    assert (manifest.collection, manifest.points, manifest.embedding_model) == (
        "c",
        3,
        "voyage-4",
    )


def test_a_damaged_file_is_refused_before_any_upload(tmp_path: Path) -> None:
    import hashlib

    path = _write(
        tmp_path, b"original bytes", hashlib.sha256(b"original bytes").hexdigest()
    )
    path.write_bytes(b"corrupted bytes")  # e.g. a truncated copy

    class _NoServer:  # restore must stop before touching Qdrant
        def __getattr__(self, name):
            raise AssertionError(f"called Qdrant: {name}")

    with pytest.raises(SnapshotError, match="SHA-256 differs"):
        snapshot.restore(path, client=_NoServer(), url="http://unused")


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        snapshot.main([])


# --- live round trip ------------------------------------------------------------

SOURCE, RESTORED = "test_snapshot_src", "test_snapshot_dst"


@pytest.fixture
def server() -> QdrantClient:
    client = QdrantClient(url=settings.QDRANT_URL)
    for name in (SOURCE, RESTORED):
        if client.collection_exists(name):
            client.delete_collection(name)
    client.create_collection(
        SOURCE,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    client.create_payload_index(SOURCE, "ciks", models.PayloadSchemaType.INTEGER)
    client.upsert(
        SOURCE,
        [
            models.PointStruct(
                id=i,
                vector=[1.0, float(i), 0.0, 0.0],
                payload={"ticker": "AAPL", "ciks": [320193]},
            )
            for i in range(3)
        ],
        wait=True,
    )
    yield client
    for name in (SOURCE, RESTORED):
        if client.collection_exists(name):
            client.delete_collection(name)


@pytest.mark.qdrant
def test_live_round_trip(server: QdrantClient, tmp_path: Path) -> None:
    path = snapshot.create(server, SOURCE, tmp_path)

    manifest = read_manifest(path)
    assert manifest.points == 3
    assert (manifest.vector_size, manifest.distance) == (4, "Cosine")
    assert server.list_snapshots(SOURCE) == []  # the server's copy was removed

    snapshot.restore(path, client=server, collection=RESTORED)

    assert server.count(RESTORED).count == 3
    [a] = server.retrieve(SOURCE, [2], with_vectors=True)
    [b] = server.retrieve(RESTORED, [2], with_vectors=True)
    assert (a.payload, a.vector) == (b.payload, b.vector)
    assert "ciks" in server.get_collection(RESTORED).payload_schema  # indexes too


@pytest.mark.qdrant
def test_live_restore_replaces_what_is_there(
    server: QdrantClient, tmp_path: Path
) -> None:
    """priority=snapshot: the file wins over a collection that drifted."""
    path = snapshot.create(server, SOURCE, tmp_path)
    server.upsert(
        SOURCE,
        [models.PointStruct(id=99, vector=[0.0, 1.0, 0.0, 0.0], payload={})],
        wait=True,
    )
    assert server.count(SOURCE).count == 4

    snapshot.restore(path, client=server)  # back into the original name

    assert server.count(SOURCE).count == 3
