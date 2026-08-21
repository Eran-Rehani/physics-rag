from __future__ import annotations

from typing import Any

from physics_rag.embeddings import PASSAGE_PREFIX, QUERY_PREFIX, E5Embedder


class _StubST:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1] * self.dimension for _ in texts]

    def get_embedding_dimension(self) -> int:
        return self.dimension


def _make_embedder(model: object) -> E5Embedder:
    # Bypass __init__ so no real model is loaded.
    embedder = object.__new__(E5Embedder)
    embedder._model = model
    embedder.batch_size = 7
    return embedder


def test_prefix_constants() -> None:
    assert QUERY_PREFIX == "query: "
    assert PASSAGE_PREFIX == "passage: "


def test_embed_documents_adds_passage_prefix() -> None:
    model = _StubST()
    embedder = _make_embedder(model)

    result = embedder.embed_documents(["Einstein solid", "אנטרופיה"])

    assert model.calls == [["passage: Einstein solid", "passage: אנטרופיה"]]
    assert result == [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]


def test_embed_query_adds_query_prefix() -> None:
    model = _StubST()
    embedder = _make_embedder(model)

    result = embedder.embed_query("entropy of Einstein solid")

    assert model.calls == [["query: entropy of Einstein solid"]]
    assert result == [0.1, 0.1, 0.1, 0.1]


def test_embed_documents_empty_returns_empty_without_calling_model() -> None:
    model = _StubST()
    embedder = _make_embedder(model)

    assert embedder.embed_documents([]) == []
    assert model.calls == []


def test_dimension_uses_new_getter() -> None:
    model = _StubST(dimension=11)
    embedder = _make_embedder(model)

    assert embedder.dimension == 11


def test_dimension_falls_back_to_legacy_getter() -> None:
    class _LegacyStubST:
        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[0.1] * 6 for _ in texts]

        def get_sentence_embedding_dimension(self) -> int:
            return 6

    embedder = _make_embedder(_LegacyStubST())

    assert embedder.dimension == 6
