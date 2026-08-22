from __future__ import annotations

from pathlib import Path

import pytest

from fakes import InMemoryStore
from physics_rag.chunking import Chunk
from physics_rag.normalize import content_hash
from physics_rag.store import ChromaStore


def make_chunk(
    text: str,
    source_path: Path,
    section_path: str = "sec",
    page: int | None = 1,
    part: int = 1,
) -> Chunk:
    digest = content_hash(text)
    return Chunk(
        text=text,
        source_path=source_path,
        filename=source_path.name,
        section_path=section_path,
        section_title="Section",
        level=1,
        page=page,
        math_fidelity="exact",
        part=part,
        n_parts=1,
        content_hash=digest,
        chunk_id=f"{digest}-{part}",
    )


def test_in_memory_store_round_trip_sorted() -> None:
    store = InMemoryStore()
    first = make_chunk("first", Path("first.tex"))
    second = make_chunk("second", Path("second.tex"))

    store.add([first, second], [[1.0, 0.0], [0.0, 1.0]])

    results = store.query([1.0, 0.0], top_k=2)

    assert [result.chunk_id for result in results] == [first.chunk_id, second.chunk_id]
    assert results[0].score == pytest.approx(1.0)


def test_in_memory_store_respects_top_k() -> None:
    store = InMemoryStore()
    first = make_chunk("first", Path("first.tex"))
    second = make_chunk("second", Path("second.tex"))

    store.add([first, second], [[1.0, 0.0], [0.0, 1.0]])

    results = store.query([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == first.chunk_id


def test_in_memory_store_query_empty_returns_empty() -> None:
    store = InMemoryStore()

    assert store.query([1.0, 0.0], top_k=3) == []


def test_in_memory_store_existing_ids() -> None:
    store = InMemoryStore()
    chunk = make_chunk("first", Path("first.tex"))

    store.add([chunk], [[1.0, 0.0]])

    assert store.existing_ids([chunk.chunk_id, "missing"]) == {chunk.chunk_id}


def test_in_memory_store_delete_by_source() -> None:
    store = InMemoryStore()
    first = make_chunk("first", Path("first.tex"))
    second = make_chunk("second", Path("second.tex"))

    store.add([first, second], [[1.0, 0.0], [0.0, 1.0]])
    store.delete_by_source("first.tex")

    remaining = store.query([0.0, 1.0], top_k=2)
    assert len(remaining) == 1
    assert remaining[0].chunk_id == second.chunk_id


def test_chroma_round_trip_is_cosine_and_upserts(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")

    store = ChromaStore(tmp_path / "chroma", collection_name="test_collection")
    chunk = make_chunk("hello chroma", Path("doc.tex"), section_path="Sec", page=3)

    store.add([chunk], [[1.0, 0.0]])
    store.add([chunk], [[1.0, 0.0]])

    assert store.count() == 1

    results = store.query([1.0, 0.0], top_k=3)
    assert results[0].chunk_id == chunk.chunk_id
    assert results[0].page == 3
    assert results[0].section_path == "Sec"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_chroma_add_collapses_duplicate_ids_within_one_batch(tmp_path: Path) -> None:
    """Byte-identical sections in two files share a content-derived id.

    Chroma accepts the same id across separate upsert calls but raises
    DuplicateIDError when one call carries it twice, which killed a real ingest.
    """
    pytest.importorskip("chromadb")

    store = ChromaStore(tmp_path / "chroma3", collection_name="dupe_ids")
    text = "identical boilerplate section"
    first = make_chunk(text, Path("a.tex"))
    second = make_chunk(text, Path("b.tex"))
    assert first.chunk_id == second.chunk_id

    store.add([first, second], [[1.0, 0.0], [1.0, 0.0]])

    assert store.count() == 1
    results = store.query([1.0, 0.0], top_k=2)
    assert results[0].filename == "a.tex"


def test_chroma_drops_none_page_metadata(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")

    store = ChromaStore(tmp_path / "chroma2", collection_name="none_page")
    chunk = make_chunk("no page here", Path("doc.tex"), section_path="Sec", page=None)

    store.add([chunk], [[0.0, 1.0]])

    results = store.query([0.0, 1.0], top_k=1)
    assert results[0].page is None
