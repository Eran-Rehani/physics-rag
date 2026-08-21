from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fakes import FakeEmbedder, InMemoryStore
from physics_rag.chunking import Chunk
from physics_rag.config import Config, default_config
from physics_rag.normalize import content_hash
from physics_rag.retrieval import Retriever, dedupe_results, format_citation
from physics_rag.store import SearchResult


def make_config(*, abstain_threshold: float, top_k: int = 6) -> Config:
    return replace(default_config(), abstain_threshold=abstain_threshold, top_k=top_k)


def make_chunk(
    text: str,
    source_path: Path,
    section_path: str = "",
    page: int | None = None,
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
        part=1,
        n_parts=1,
        content_hash=digest,
        chunk_id=f"{digest}-1",
    )


def make_result(
    text: str,
    *,
    filename: str = "file.tex",
    section_path: str = "",
    page: int | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=content_hash(text),
        text=text,
        score=0.8,
        filename=filename,
        section_path=section_path,
        page=page,
        math_fidelity="exact",
        source_path="file.tex",
    )


def test_retrieve_returns_results_and_confidence() -> None:
    embedder = FakeEmbedder(dimension=16)
    store = InMemoryStore()
    chunk = make_chunk("answer text", Path("notes.tex"), section_path="sec 1", page=2)

    store.add([chunk], [embedder.embed_documents(["answer text"])[0]])
    retriever = Retriever(embedder, store, make_config(abstain_threshold=2.0))
    result = retriever.retrieve("question")

    assert result.question == "question"
    assert len(result.results) == 1
    assert result.results[0].text == "answer text"
    top_score = store.query(embedder.embed_query("question"), top_k=1)[0].score
    assert result.confidence == pytest.approx(top_score)
    assert result.abstain is True


def test_retrieve_does_not_abstain_when_score_above_threshold() -> None:
    embedder = FakeEmbedder(dimension=16)
    store = InMemoryStore()
    chunk = make_chunk("answer text", Path("notes.tex"), section_path="sec 1", page=2)

    store.add([chunk], [embedder.embed_documents(["answer text"])[0]])
    retriever = Retriever(embedder, store, make_config(abstain_threshold=-1.0))

    assert retriever.retrieve("question").abstain is False


def test_retrieve_empty_store_abstains() -> None:
    embedder = FakeEmbedder(dimension=8)
    store = InMemoryStore()
    retriever = Retriever(embedder, store, make_config(abstain_threshold=0.35))

    result = retriever.retrieve("anything")

    assert result.results == []
    assert result.confidence == 0.0
    assert result.abstain is True


def test_retrieve_defaults_to_config_top_k() -> None:
    embedder = FakeEmbedder(dimension=8)
    store = InMemoryStore()
    chunks = [make_chunk(f"chunk {index}", Path(f"chunk-{index}.tex")) for index in range(3)]
    store.add(chunks, [embedder.embed_documents([chunk.text])[0] for chunk in chunks])

    retriever = Retriever(embedder, store, make_config(abstain_threshold=-1.0, top_k=2))

    assert len(retriever.retrieve("question").results) == 2


def test_retriever_embeds_the_question_as_a_query() -> None:
    embedder = FakeEmbedder(dimension=8)
    store = InMemoryStore()
    retriever = Retriever(embedder, store, make_config(abstain_threshold=0.35))

    retriever.retrieve("what is entropy")

    assert embedder.seen_queries == ["what is entropy"]


def test_format_citation_section_page_and_bare() -> None:
    section_result = make_result("text", filename="notes.tex", section_path="Waves", page=4)
    assert format_citation(section_result) == "[notes.tex, Waves]"

    page_result = make_result("text", filename="notes.tex", section_path="", page=4)
    assert format_citation(page_result) == "[notes.tex, p. 4]"

    bare_result = make_result("text", filename="notes.tex", section_path="", page=None)
    assert format_citation(bare_result) == "[notes.tex]"


def test_dedupe_results_drops_identical_text() -> None:
    first = make_result("duplicate", filename="first.tex")
    second = make_result("duplicate", filename="second.tex")
    unique = make_result("unique", filename="third.tex")

    assert dedupe_results([first, second, unique]) == [first, unique]
