"""Tests for Voyage embedding. ``-m voyage`` runs the ones that call the API."""

from types import SimpleNamespace

import pytest

from finai.config import settings
from finai.ingestion import embed


def test_batches_respect_the_token_budget() -> None:
    texts = [f"t{i}" for i in range(7)]

    out = list(embed.batches(texts, [40] * 7, max_tokens=100, max_texts=1_000))

    assert out == [["t0", "t1"], ["t2", "t3"], ["t4", "t5"], ["t6"]]


def test_batches_respect_the_text_limit() -> None:
    texts = [f"t{i}" for i in range(5)]

    out = list(embed.batches(texts, [1] * 5, max_tokens=1_000, max_texts=2))

    assert out == [["t0", "t1"], ["t2", "t3"], ["t4"]]


def test_an_oversized_text_gets_a_batch_of_its_own() -> None:
    out = list(embed.batches(["a", "big", "b"], [10, 500, 10], max_tokens=100))

    assert out == [["a"], ["big"], ["b"]]


def test_voyage_client_is_shared_and_retries() -> None:
    embed.voyage_client.cache_clear()

    first, second = embed.voyage_client(), embed.voyage_client()

    assert first is second
    assert first.max_retries == embed.MAX_RETRIES > 0


class _FakeVoyage:
    """Records embed calls; returns a vector encoding each text's position."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def count_tokens(self, texts: list[str], model: str) -> int:
        return 60_000 if texts[0].startswith("long") else 10

    def embed(self, texts, model, input_type, output_dimension):
        self.calls.append(
            {
                "n": len(texts),
                "model": model,
                "input_type": input_type,
                "dim": output_dimension,
            }
        )
        return SimpleNamespace(embeddings=[[float(t.split()[-1])] for t in texts])


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeVoyage:
    client = _FakeVoyage()
    monkeypatch.setattr(embed, "voyage_client", lambda: client)
    return client


def test_documents_keep_their_order_across_batches(fake: _FakeVoyage) -> None:
    texts = [f"long chunk {i}" for i in range(3)] + [f"short {i}" for i in range(3, 6)]

    vectors = embed.embed_documents(texts)

    assert vectors == [[float(i)] for i in range(6)]
    assert len(fake.calls) > 1  # 60K-token texts forced several requests
    assert {c["input_type"] for c in fake.calls} == {"document"}
    assert {c["model"] for c in fake.calls} == {settings.EMBEDDING_MODEL}
    assert {c["dim"] for c in fake.calls} == {settings.EMBEDDING_DIM}


def test_query_uses_the_query_input_type(fake: _FakeVoyage) -> None:
    assert embed.embed_query("supply risk 7") == [7.0]
    assert fake.calls == [
        {
            "n": 1,
            "model": settings.EMBEDDING_MODEL,
            "input_type": "query",
            "dim": settings.EMBEDDING_DIM,
        }
    ]


@pytest.mark.voyage
def test_live_embeddings_have_the_configured_size() -> None:
    embed.voyage_client.cache_clear()

    docs = embed.embed_documents(["Apple obtains components from single sources."])
    query = embed.embed_query("What are Apple's supply chain risks?")

    assert len(docs) == 1
    assert len(docs[0]) == len(query) == settings.EMBEDDING_DIM
