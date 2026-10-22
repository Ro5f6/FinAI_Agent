"""Save the 10-K collection to a file, and restore it anywhere (1.3.33).

    python -m finai.ingestion.snapshot create
    python -m finai.ingestion.snapshot restore data/snapshots/<name>.snapshot \\
        [--url http://localhost:6333] [--collection sec_filings]

Embedding the S&P 500 took 40 minutes of Voyage calls; restoring a snapshot
takes seconds and no tokens. The cluster's Qdrant volume is deleted after
every AWS session (4.2.18), so the copy that survives is the file kept here,
in ``SNAPSHOT_DIR``, restored into the fresh cluster each time (4.2.14).

How, verified against qdrant-client 1.19.1 and Qdrant v1.19.1
(docs/CONTEXT.md):

* **Create:** ``create_snapshot(collection, wait=True)`` writes the snapshot
  inside the server (3.8 s, 480 MB for 47,233 points) and returns its name,
  size and ``checksum``, the file's SHA-256.
* **Download:** ``GET /collections/{c}/snapshots/{name}`` streams the file;
  it is written to disk in 1 MB pieces and its SHA-256 compared with the
  server's. The server's copy is then deleted, so snapshots don't pile up
  inside Qdrant's storage.
* **Restore:** ``POST /collections/{c}/snapshots/upload`` with the file as
  multipart field ``snapshot``. ``recover_snapshot`` can't be used: it takes
  a location the *server* fetches (a URL or a path on its own disk), and the
  file is on this machine. ``priority=snapshot`` makes the file's data win
  over anything already in the collection; ``checksum`` makes the server
  reject a damaged upload. A missing collection is created.
* **Check:** a manifest written next to the snapshot records the point count,
  vector size and distance, embedding model and checksum; a restore is only
  reported done when the restored collection matches it.
"""

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import httpx
from qdrant_client import QdrantClient

from finai.config import settings
from finai.ingestion.store import qdrant_client

_PIECE = 1 << 20  # 1 MB


class SnapshotError(RuntimeError):
    """A snapshot didn't arrive intact or didn't restore what it holds."""


class Manifest(NamedTuple):
    """What a snapshot file holds, written beside it as ``<name>.json``."""

    collection: str
    snapshot: str
    sha256: str
    bytes: int
    points: int
    vector_size: int
    distance: str
    embedding_model: str
    created_at: str


def manifest_path(snapshot_file: Path) -> Path:
    return snapshot_file.with_suffix(".json")


def read_manifest(snapshot_file: Path) -> Manifest:
    return Manifest(**json.loads(manifest_path(snapshot_file).read_text()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while piece := f.read(_PIECE):
            digest.update(piece)
    return digest.hexdigest()


def _vectors(client: QdrantClient, collection: str) -> tuple[int, str]:
    config = client.get_collection(collection).config.params.vectors
    return config.size, str(config.distance.value)


def create(
    client: QdrantClient | None = None,
    collection: str | None = None,
    directory: Path | None = None,
    url: str | None = None,
) -> Path:
    """Snapshot ``collection``, download it to ``directory``, return the file.

    Raises:
        SnapshotError: the download's SHA-256 differs from the server's.
    """
    client = client or qdrant_client()
    collection = collection or settings.COLLECTION_NAME
    directory = directory or settings.SNAPSHOT_DIR
    url = url or settings.QDRANT_URL
    directory.mkdir(parents=True, exist_ok=True)

    points = client.count(collection, exact=True).count
    vector_size, distance = _vectors(client, collection)
    description = client.create_snapshot(collection, wait=True)

    target = directory / description.name
    partial = target.with_suffix(".partial")
    digest = hashlib.sha256()
    try:
        with (
            httpx.stream(
                "GET",
                f"{url}/collections/{collection}/snapshots/{description.name}",
                timeout=httpx.Timeout(30.0, read=None),
            ) as response,
            partial.open("wb") as f,
        ):
            response.raise_for_status()
            for piece in response.iter_bytes(_PIECE):
                digest.update(piece)
                f.write(piece)
    finally:
        # The file is the copy that matters; the server's would only take
        # space inside Qdrant's storage.
        client.delete_snapshot(collection, description.name, wait=True)

    if digest.hexdigest() != description.checksum:
        partial.unlink()
        raise SnapshotError(
            f"{description.name}: downloaded SHA-256 {digest.hexdigest()} "
            f"!= server's {description.checksum}"
        )
    partial.rename(target)

    manifest = Manifest(
        collection=collection,
        snapshot=description.name,
        sha256=description.checksum,
        bytes=target.stat().st_size,
        points=points,
        vector_size=vector_size,
        distance=distance,
        embedding_model=settings.EMBEDDING_MODEL,
        created_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    manifest_path(target).write_text(json.dumps(manifest._asdict(), indent=2) + "\n")
    return target


def restore(
    snapshot_file: Path,
    client: QdrantClient | None = None,
    collection: str | None = None,
    url: str | None = None,
) -> Manifest:
    """Upload ``snapshot_file`` into ``collection`` and check it matches.

    Raises:
        SnapshotError: the file differs from its manifest, or the restored
            collection doesn't match it.
    """
    client = client or qdrant_client()
    url = url or settings.QDRANT_URL
    manifest = read_manifest(snapshot_file)
    collection = collection or manifest.collection

    if _sha256(snapshot_file) != manifest.sha256:
        raise SnapshotError(f"{snapshot_file}: SHA-256 differs from its manifest")

    with snapshot_file.open("rb") as f:
        response = httpx.post(
            f"{url}/collections/{collection}/snapshots/upload",
            params={
                "wait": "true",
                "priority": "snapshot",
                "checksum": manifest.sha256,
            },
            files={"snapshot": (snapshot_file.name, f, "application/octet-stream")},
            timeout=httpx.Timeout(30.0, read=None, write=None),
        )
    response.raise_for_status()

    points = client.count(collection, exact=True).count
    vectors = _vectors(client, collection)
    if (points, vectors) != (
        manifest.points,
        (manifest.vector_size, manifest.distance),
    ):
        raise SnapshotError(
            f"{collection} after restore: {points} points, vectors {vectors}; "
            f"snapshot holds {manifest.points}, "
            f"({manifest.vector_size}, {manifest.distance!r})"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m finai.ingestion.snapshot",
        description="Save the 10-K collection to a file, or restore it from one.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create", help="snapshot the collection into SNAPSHOT_DIR")
    restore_cmd = commands.add_parser("restore", help="upload a snapshot into a Qdrant")
    restore_cmd.add_argument("file", type=Path, help="a .snapshot file with its .json")
    restore_cmd.add_argument(
        "--url", help="Qdrant to restore into (default QDRANT_URL)"
    )
    restore_cmd.add_argument(
        "--collection", help="name to restore as (default: the original)"
    )
    args = parser.parse_args(argv)

    started = time.monotonic()
    if args.command == "create":
        path = create()
        manifest = read_manifest(path)
        print(
            f"{path} ({manifest.bytes / 1e6:.0f} MB, {manifest.points} points) "
            f"in {time.monotonic() - started:.1f} s"
        )
    else:
        url = args.url or settings.QDRANT_URL
        manifest = restore(
            args.file, client=QdrantClient(url=url), collection=args.collection, url=url
        )
        print(
            f"restored {manifest.points} points into "
            f"{args.collection or manifest.collection} at {url} "
            f"in {time.monotonic() - started:.1f} s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
