from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fakes import FakeEmbedder, InMemoryStore
from physics_rag.answer import (
    ABSTAIN_MESSAGE,
    AnswerService,
    build_context,
    build_prompt,
    extract_cited_labels,
    resolve_numeric_citations,
)
from physics_rag.chunking import Chunk
from physics_rag.config import Config, default_config
from physics_rag.normalize import content_hash
from physics_rag.retrieval import Retriever
from physics_rag.store import SearchResult


class FakeGenerator:
    def __init__(self, reply: str = "all good") -> None:
        self.reply = reply
        self.calls = 0
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.systems.append(system)
        return self.reply


def _config(threshold: float) -> Config:
    return replace(default_config(), abstain_threshold=threshold)


def _chunk(
    text: str,
    *,
    filename: str = "notes.tex",
    section_path: str = "",
    page: int | None = None,
    math_fidelity: str = "exact",
) -> Chunk:
    digest = content_hash(text)
    return Chunk(
        text=text,
        source_path=Path(f"/tmp/{filename}"),
        filename=filename,
        section_path=section_path,
        section_title=section_path.split(" > ")[-1],
        level=1,
        page=page,
        math_fidelity=math_fidelity,
        part=1,
        n_parts=1,
        content_hash=digest,
        chunk_id=f"{digest}-1",
    )


def _store_with(embedder: FakeEmbedder, chunks: list[Chunk]) -> InMemoryStore:
    store = InMemoryStore()
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return store


def _result(
    text: str,
    *,
    filename: str,
    section_path: str = "",
    page: int | None = None,
    math_fidelity: str = "exact",
) -> SearchResult:
    return SearchResult(
        chunk_id=content_hash(text),
        text=text,
        score=0.9,
        filename=filename,
        section_path=section_path,
        page=page,
        math_fidelity=math_fidelity,
        source_path=f"/tmp/{filename}",
    )


def test_empty_store_abstains_without_calling_generator() -> None:
    config = _config(0.35)
    retriever = Retriever(FakeEmbedder(), InMemoryStore(), config)
    generator = FakeGenerator("should not be called")

    answer = AnswerService(retriever, generator, config).ask("What is entropy?")

    assert answer.abstained is True
    assert answer.text == ABSTAIN_MESSAGE
    assert answer.citations == []
    assert generator.calls == 0


def test_low_confidence_abstains_without_calling_generator() -> None:
    config = _config(1.01)
    embedder = FakeEmbedder()
    store = _store_with(embedder, [_chunk("entropy is disorder")])
    generator = FakeGenerator("should not be called")

    answer = AnswerService(Retriever(embedder, store, config), generator, config).ask("entropy?")

    assert answer.abstained is True
    assert answer.text == ABSTAIN_MESSAGE
    # A low-confidence retrieval must never cost a generation.
    assert generator.calls == 0


def test_high_confidence_calls_generator_and_cites() -> None:
    config = _config(-1.0)
    embedder = FakeEmbedder()
    chunk = _chunk(r"Entropy is $S=k_{B}\ln\Omega$.", section_path="Week 2 > Entropy")
    store = _store_with(embedder, [chunk])
    generator = FakeGenerator("The entropy is [notes.tex, Week 2 > Entropy].")

    answer = AnswerService(Retriever(embedder, store, config), generator, config).ask("entropy?")

    assert answer.abstained is False
    assert answer.citations == ["[notes.tex, Week 2 > Entropy]"]
    assert generator.calls == 1
    assert generator.systems[0] is not None


def test_prompt_contains_numbered_blocks_labels_and_math_markers() -> None:
    config = _config(-1.0)
    embedder = FakeEmbedder()
    chunks = [
        _chunk("Exact math $E=mc^2$.", filename="a.tex", section_path="One"),
        _chunk(
            "Degraded equation E 2 = p2 c2.",
            filename="b.pdf",
            section_path="Two",
            math_fidelity="degraded",
        ),
    ]
    store = _store_with(embedder, chunks)
    generator = FakeGenerator("answer")

    AnswerService(Retriever(embedder, store, config), generator, config).ask("What is energy?")

    prompt = generator.prompts[0]
    assert "SOURCE 1" in prompt and "SOURCE 2" in prompt
    # Bare numeric markers must not look like citations or the model copies them.
    assert "[1]" not in prompt and "[2]" not in prompt
    assert "(math: exact)" in prompt
    assert "(math: degraded)" in prompt
    assert "[a.tex, One]" in prompt
    assert "[b.pdf, Two]" in prompt
    assert "What is energy?" in prompt


def test_model_reply_not_found_marks_answer_abstained() -> None:
    config = _config(-1.0)
    embedder = FakeEmbedder()
    store = _store_with(embedder, [_chunk("some content", section_path="Sec")])
    generator = FakeGenerator("Not found in corpus.")

    answer = AnswerService(Retriever(embedder, store, config), generator, config).ask(
        "dark energy?"
    )

    assert answer.abstained is True
    assert answer.text == ABSTAIN_MESSAGE
    assert answer.citations == []
    assert generator.calls == 1


def test_extract_cited_labels_unicode_ordered_and_deduped() -> None:
    text = "See [a.tex] and [Lec_03.tex, שבוע 2 > אנטרופיה] and [a.tex] again."

    assert extract_cited_labels(text) == ["[a.tex]", "[Lec_03.tex, שבוע 2 > אנטרופיה]"]


def test_build_context_marks_degraded_sources() -> None:
    context = build_context(
        [
            _result("exact chunk", filename="a.tex", section_path="One"),
            _result("degraded chunk", filename="b.pdf", page=2, math_fidelity="degraded"),
        ]
    )

    assert "SOURCE 1 -- cite as [a.tex, One] (math: exact)" in context
    assert "SOURCE 2 -- cite as [b.pdf, p. 2] (math: degraded)" in context
    assert "exact chunk" in context
    assert "degraded chunk" in context


def test_build_prompt_includes_question_and_sources() -> None:
    prompt = build_prompt("Why?", [_result("body", filename="a.tex", section_path="S")])

    assert "Sources:" in prompt
    assert "[a.tex, S]" in prompt
    assert prompt.rstrip().endswith("Question: Why?")


def test_resolve_numeric_citations_rewrites_bare_numbers() -> None:
    results = [
        _result("a", filename="a.tex", section_path="One"),
        _result("b", filename="b.pdf", page=7, math_fidelity="degraded"),
    ]

    text = "Claim one [1] and claim two [SOURCE 2]. Out of range [9] stays."

    assert resolve_numeric_citations(text, results) == (
        "Claim one [a.tex, One] and claim two [b.pdf, p. 7]. Out of range [9] stays."
    )


def test_numeric_citation_is_resolved_end_to_end() -> None:
    config = _config(-1.0)
    embedder = FakeEmbedder()
    chunk = _chunk("Friedmann content here for the test.", filename="ryden.pdf", section_path="4.2")
    store = _store_with(embedder, [chunk])
    # A small model citing the source number instead of the label must still
    # produce a usable [filename, section] citation.
    generator = FakeGenerator("The equation binds curvature and density [1].")

    answer = AnswerService(Retriever(embedder, store, config), generator, config).ask("friedmann?")

    assert "[ryden.pdf, 4.2]" in answer.text
    assert extract_cited_labels(answer.text) == ["[ryden.pdf, 4.2]"]


def test_with_config_reuses_retriever_and_generator() -> None:
    config = _config(-1.0)
    embedder = FakeEmbedder()
    store = _store_with(embedder, [_chunk("body text here", section_path="Sec")])
    generator = FakeGenerator("answer [notes.tex, Sec]")
    service = AnswerService(Retriever(embedder, store, config), generator, config)

    # Raising the threshold above any score must flip the same service to abstaining.
    strict = service.with_config(_config(1.01))

    assert strict.retriever is service.retriever
    assert strict.generator is service.generator
    assert strict.ask("q").abstained is True
    assert generator.calls == 0
