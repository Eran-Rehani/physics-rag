from __future__ import annotations

from dataclasses import dataclass

from physics_rag.config import Config
from physics_rag.embeddings import Embedder
from physics_rag.store import SearchResult, VectorStore


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    question: str
    results: list[SearchResult]
    confidence: float
    abstain: bool


def format_citation(result: SearchResult) -> str:
    """Render a citation as [filename, section] / [filename, p. N] / [filename]."""
    if result.section_path:
        return f"[{result.filename}, {result.section_path}]"
    if result.page is not None:
        return f"[{result.filename}, p. {result.page}]"
    return f"[{result.filename}]"


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    """Drop results whose text repeats an earlier one, preserving order."""
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for result in results:
        if result.text in seen:
            continue
        seen.add(result.text)
        deduped.append(result)
    return deduped


class Retriever:
    """Embeds a question, queries the store, and decides whether to abstain."""

    def __init__(self, embedder: Embedder, store: VectorStore, config: Config) -> None:
        self._embedder = embedder
        self._store = store
        self._config = config

    def retrieve(self, question: str, *, top_k: int | None = None) -> RetrievalResult:
        embedding = self._embedder.embed_query(question)
        k = self._config.top_k if top_k is None else top_k
        results = self._store.query(embedding, top_k=k)
        confidence = results[0].score if results else 0.0
        abstain = confidence < self._config.abstain_threshold
        return RetrievalResult(
            question=question,
            results=results,
            confidence=confidence,
            abstain=abstain,
        )
