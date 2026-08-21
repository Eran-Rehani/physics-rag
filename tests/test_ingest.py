from __future__ import annotations

from pathlib import Path
from typing import Any

from fakes import FakeEmbedder, InMemoryStore
from physics_rag.chunking import chunk_document, chunk_embedding_text
from physics_rag.config import default_config
from physics_rag.ingest import IngestState, dedupe_sources, discover_files, ingest
from physics_rag.parsers.base import ParsedDocument
from physics_rag.parsers.tex import TexParser

# Bodies must exceed Config.min_chunk_chars (80) or the chunker correctly drops them.
BODY = (
    "Some body text for chunking that is comfortably longer than "
    "the minimum chunk size so it survives."
)
BODY_BAD = (
    "Bad body that is comfortably longer than the minimum chunk size so it would survive chunking."
)
BODY_GOOD = (
    "Good body that is comfortably longer than the minimum chunk size so it survives chunking fine."
)
BODY_SHORT = (
    "Short body that is nonetheless comfortably longer than the minimum chunk size so it survives."
)
BODY_CHANGED = (
    "A completely different body, also comfortably longer than "
    "the minimum chunk size, which changes the hash."
)
BODY_SECTION = (
    "Some section body goes here and it is comfortably longer "
    "than the minimum chunk size threshold."
)


def write_tex(path: Path, body: str) -> None:
    path.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{Intro}\n"
        f"{body}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )


def _flatten_texts(texts: Any) -> list[str]:
    flat: list[str] = []
    for item in texts:
        if isinstance(item, str):
            flat.append(item)
        else:
            flat.extend(item)
    return flat


def test_discover_files_filters(tmp_path: Path) -> None:
    root = tmp_path
    (root / "main.tex").write_text("%", encoding="utf-8")
    (root / "paper.pdf").write_text("%PDF", encoding="utf-8")
    (root / "main.aux").write_text("aux", encoding="utf-8")

    (root / "Docs").mkdir()
    (root / "Docs" / "note.tex").write_text("%", encoding="utf-8")

    (root / ".git").mkdir()
    (root / ".git" / "config.tex").write_text("%", encoding="utf-8")

    (root / ".hidden.tex").write_text("%", encoding="utf-8")

    files = discover_files(root, default_config())

    assert files == [root / "main.tex", root / "paper.pdf"]


def test_dedupe_sources_drops_pdf_next_to_tex(tmp_path: Path) -> None:
    a, b, c, d = (tmp_path / name for name in ("a", "b", "c", "d"))
    for directory in (a, b, c, d):
        directory.mkdir()

    sol_tex = a / "Sol_05.tex"
    sol_pdf = a / "Sol_05.pdf"
    sol_tex.write_text("%", encoding="utf-8")
    sol_pdf.write_text("%PDF", encoding="utf-8")

    lecture_pdf = b / "Lecture.pdf"
    lecture_pdf.write_text("%PDF", encoding="utf-8")

    same_tex = c / "same_stem.tex"
    same_tex.write_text("%", encoding="utf-8")
    same_pdf = d / "same_stem.pdf"
    same_pdf.write_text("%PDF", encoding="utf-8")

    kept, dropped = dedupe_sources([sol_tex, sol_pdf, lecture_pdf, same_tex, same_pdf])

    assert kept == [sol_tex, lecture_pdf, same_tex, same_pdf]
    assert dropped == [sol_pdf]


def test_ingest_populates_store(tmp_path: Path) -> None:
    root = tmp_path
    write_tex(
        root / "main.tex",
        BODY,
    )

    embedder = FakeEmbedder(dimension=8)
    store = InMemoryStore()
    state = IngestState(root / "ingest-state.json")

    stats = ingest(
        root,
        config=default_config(),
        embedder=embedder,
        store=store,
        parsers=[TexParser()],
        state=state,
    )

    assert stats.files_seen == 1
    assert stats.files_parsed == 1
    assert stats.chunks_added == store.count()
    assert store.count() > 0


def test_second_ingest_skips_unchanged(tmp_path: Path) -> None:
    root = tmp_path
    write_tex(
        root / "main.tex",
        BODY,
    )

    config = default_config()
    store = InMemoryStore()
    state = IngestState(root / "ingest-state.json")

    first = ingest(
        root,
        config=config,
        embedder=FakeEmbedder(dimension=8),
        store=store,
        parsers=[TexParser()],
        state=state,
    )
    first_count = store.count()

    second = ingest(
        root,
        config=config,
        embedder=FakeEmbedder(dimension=8),
        store=store,
        parsers=[TexParser()],
        state=state,
    )

    assert first.files_skipped_unchanged == 0
    assert second.files_skipped_unchanged == 1
    assert second.chunks_added == 0
    assert store.count() == first_count


def test_force_reingests(tmp_path: Path) -> None:
    root = tmp_path
    write_tex(
        root / "main.tex",
        BODY,
    )

    config = default_config()
    store = InMemoryStore()
    state = IngestState(root / "ingest-state.json")

    ingest(
        root,
        config=config,
        embedder=FakeEmbedder(dimension=8),
        store=store,
        parsers=[TexParser()],
        state=state,
    )
    first_count = store.count()

    second = ingest(
        root,
        config=config,
        embedder=FakeEmbedder(dimension=8),
        store=store,
        parsers=[TexParser()],
        state=state,
        force=True,
    )

    assert second.files_skipped_unchanged == 0
    assert second.files_parsed == 1
    assert second.chunks_added > 0
    assert store.count() == first_count


def test_touching_file_causes_reingest_and_old_ids_removed(tmp_path: Path) -> None:
    root = tmp_path
    tex = root / "main.tex"
    write_tex(
        tex,
        BODY_SHORT,
    )

    config = default_config()
    store = InMemoryStore()
    state = IngestState(root / "ingest-state.json")

    ingest(
        root,
        config=config,
        embedder=FakeEmbedder(dimension=8),
        store=store,
        parsers=[TexParser()],
        state=state,
    )

    chunk_ids = state.files[str(tex)]["chunk_ids"]
    assert isinstance(chunk_ids, list)
    old_ids = [str(chunk_id) for chunk_id in chunk_ids]
    assert old_ids

    write_tex(
        tex,
        BODY_CHANGED,
    )

    second = ingest(
        root,
        config=config,
        embedder=FakeEmbedder(dimension=8),
        store=store,
        parsers=[TexParser()],
        state=state,
    )

    assert second.files_skipped_unchanged == 0
    assert second.files_parsed == 1
    assert store.existing_ids(old_ids) == set()


def test_parser_exception_is_counted_and_does_not_abort(tmp_path: Path) -> None:
    root = tmp_path
    write_tex(
        root / "bad.tex",
        BODY_BAD,
    )
    write_tex(
        root / "good.tex",
        BODY_GOOD,
    )

    class BoomParser:
        def can_parse(self, path: Path) -> bool:
            return path.name == "bad.tex"

        def parse(self, path: Path) -> ParsedDocument:
            raise RuntimeError("boom")

    store = InMemoryStore()
    messages: list[str] = []

    stats = ingest(
        root,
        config=default_config(),
        embedder=FakeEmbedder(dimension=8),
        store=store,
        parsers=[BoomParser(), TexParser()],
        progress=messages.append,
    )

    assert stats.files_seen == 2
    assert stats.files_failed == 1
    assert stats.files_parsed == 1
    assert store.count() > 0
    assert any("failed" in message for message in messages)


def test_ingest_state_roundtrip_and_corrupt(tmp_path: Path) -> None:
    root = tmp_path
    state_path = root / "state.json"
    tex = root / "file.tex"
    tex.write_text("%", encoding="utf-8")

    state = IngestState(state_path)
    state.record(tex, ["id-1", "id-2"])
    state.save()

    assert state.is_unchanged(tex)

    loaded = IngestState.load(state_path)
    assert loaded.files == state.files
    assert loaded.is_unchanged(tex)

    state_path.write_text("{corrupt", encoding="utf-8")
    broken = IngestState.load(state_path)
    assert broken.files == {}
    assert not broken.is_unchanged(tex)


def test_embedder_receives_chunk_embedding_text(tmp_path: Path) -> None:
    root = tmp_path
    tex = root / "main.tex"
    write_tex(
        tex,
        BODY_SECTION,
    )

    config = default_config()
    expected_chunks = chunk_document(TexParser().parse(tex), config)
    expected_texts = [chunk_embedding_text(chunk) for chunk in expected_chunks]
    assert expected_texts

    embedder = FakeEmbedder(dimension=8)

    ingest(
        root,
        config=config,
        embedder=embedder,
        store=InMemoryStore(),
        parsers=[TexParser()],
    )

    seen = _flatten_texts(embedder.seen_documents)

    for expected in expected_texts:
        assert expected in seen

    # The bare chunk text must NOT be what gets embedded when a section path exists.
    for chunk in expected_chunks:
        if chunk_embedding_text(chunk) != chunk.text:
            assert chunk.text not in seen
