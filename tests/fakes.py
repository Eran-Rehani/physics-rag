"""Deterministic fakes so the test suite needs neither torch nor chromadb."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from physics_rag.chunking import Chunk
from physics_rag.store import SearchResult


class FakeEmbedder:
    """Hash-based Embedder. Records the exact strings it was handed."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.seen_documents: list[str] = []
        self.seen_queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [self._embed(text) for text in texts]
        self.seen_documents.extend(texts)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        self.seen_queries.append(text)
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(digest[i % len(digest)] - 127.5) / 127.5 for i in range(self.dimension)]
        return self._normalize(raw)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class InMemoryStore:
    """VectorStore backed by a dict, with cosine similarity computed by hand."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._items[chunk.chunk_id] = (chunk, list(embedding))

    def query(self, embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        if not self._items or top_k <= 0:
            return []
        query_vector = list(embedding)
        scored = [
            (self._cosine_similarity(query_vector, candidate), chunk)
            for chunk, candidate in self._items.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self._to_result(score, chunk) for score, chunk in scored[:top_k]]

    def count(self) -> int:
        return len(self._items)

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        return {chunk_id for chunk_id in ids if chunk_id in self._items}

    def delete_by_source(self, source_path: str) -> None:
        self._items = {
            chunk_id: item
            for chunk_id, item in self._items.items()
            if str(item[0].source_path) != source_path
        }

    @staticmethod
    def _to_result(score: float, chunk: Chunk) -> SearchResult:
        return SearchResult(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=score,
            filename=chunk.filename,
            section_path=chunk.section_path,
            page=chunk.page,
            math_fidelity=chunk.math_fidelity,
            source_path=str(chunk.source_path),
        )

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (left_norm * right_norm)))
