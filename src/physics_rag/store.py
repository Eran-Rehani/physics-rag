from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from physics_rag.chunking import Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    text: str
    score: float
    filename: str
    section_path: str
    page: int | None
    math_fidelity: str
    source_path: str


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def query(self, embedding: Sequence[float], top_k: int) -> list[SearchResult]: ...

    def count(self) -> int: ...

    def existing_ids(self, ids: Sequence[str]) -> set[str]: ...

    def delete_by_source(self, source_path: str) -> None: ...


class ChromaStore:
    """File-backed Chroma collection, explicitly configured for cosine space."""

    def __init__(self, persist_dir: Path, collection_name: str = "physics") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # Chroma defaults to L2. Without this the scores would not be cosine
        # similarities and the abstain threshold would be meaningless.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _chunk_metadata(chunk: Chunk) -> dict[str, object]:
        metadata: dict[str, object] = {
            "filename": chunk.filename,
            "section_path": chunk.section_path,
            "section_title": chunk.section_title,
            "page": chunk.page,
            "math_fidelity": chunk.math_fidelity,
            "part": chunk.part,
            "n_parts": chunk.n_parts,
            "level": chunk.level,
            "source_path": str(chunk.source_path),
            "content_hash": chunk.content_hash,
        }
        # Chroma rejects None metadata values, so drop them entirely.
        return {key: value for key, value in metadata.items() if value is not None}

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return

        # chunk_id is content-derived, so byte-identical sections share an id by
        # design -- that is what makes re-ingest idempotent. Chroma tolerates the
        # same id across upsert calls but rejects it twice within one call, so
        # collapse repeats here, keeping the first occurrence.
        seen: set[str] = set()
        ids: list[str] = []
        documents: list[str] = []
        vectors: list[list[float]] = []
        metadatas: list[dict[str, object]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            vectors.append(list(embedding))
            metadatas.append(self._chunk_metadata(chunk))

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )

    def query(self, embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids_rows = result.get("ids") or []
        ids = ids_rows[0] if ids_rows else []
        if not ids:
            return []

        documents_rows = result.get("documents") or []
        documents = documents_rows[0] if documents_rows else []
        metadatas_rows = result.get("metadatas") or []
        metadatas = metadatas_rows[0] if metadatas_rows else []
        distances_rows = result.get("distances") or []
        distances = distances_rows[0] if distances_rows else []

        results: list[SearchResult] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            distance = float(distances[index]) if index < len(distances) else 0.0
            score = max(0.0, min(1.0, 1.0 - distance))
            page = metadata.get("page")
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=documents[index] if index < len(documents) else "",
                    score=score,
                    filename=str(metadata.get("filename", "")),
                    section_path=str(metadata.get("section_path", "")),
                    page=int(page) if page is not None else None,
                    math_fidelity=str(metadata.get("math_fidelity", "")),
                    source_path=str(metadata.get("source_path", "")),
                )
            )
        return results

    def count(self) -> int:
        return int(self._collection.count())

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        try:
            response = self._collection.get(ids=list(ids), include=[])
            return set(response.get("ids", []))
        except Exception:
            return set()

    def delete_by_source(self, source_path: str) -> None:
        self._collection.delete(where={"source_path": source_path})
