from __future__ import annotations

import re
from pathlib import Path

from physics_rag.chunking import (
    Chunk,
    chunk_document,
    chunk_documents,
    chunk_embedding_text,
    find_math_spans,
    is_inside_math,
)
from physics_rag.config import Config
from physics_rag.parsers.base import MathFidelity, ParsedDocument, Section


def _make_config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "corpus_root": Path("/tmp/corpus"),
        "staging_dir": Path("/tmp/staging"),
        "chroma_dir": Path("/tmp/chroma"),
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def _make_document(sections: list[Section]) -> ParsedDocument:
    return ParsedDocument(
        source_path=Path("/tmp/test.tex"),
        sections=sections,
        math_fidelity=MathFidelity.EXACT,
    )


def _section(
    body: str,
    title: str = "Section",
    level: int = 2,
    page: int | None = None,
    path: tuple[str, ...] = ("Section",),
) -> Section:
    return Section(title=title, level=level, body=body, page=page, path=path)


def _unescaped_dollar_count(text: str) -> int:
    total = 0
    for i, char in enumerate(text):
        if char != "$":
            continue
        backslashes = 0
        j = i - 1
        while j >= 0 and text[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes % 2 == 0:
            total += 1
    return total


def test_short_section_is_one_chunk() -> None:
    body = "word " * 30
    section = _section(body, title="Intro", page=1, path=("Week 1", "Intro"))

    chunks = chunk_document(_make_document([section]), _make_config())

    assert len(chunks) == 1
    assert chunks[0].part == 1
    assert chunks[0].n_parts == 1
    assert chunks[0].section_path == "Week 1 > Intro"
    assert chunks[0].page == 1
    assert chunks[0].math_fidelity == "exact"


def test_find_math_spans_recognises_delimiters() -> None:
    text = r"$x$ \[y\] \begin{align} z \end{align} \$notmath"
    spans = find_math_spans(text)

    assert len(spans) == 3
    assert text[spans[0][0] : spans[0][1]] == "$x$"
    assert text[spans[1][0] : spans[1][1]] == r"\[y\]"
    assert text[spans[2][0] : spans[2][1]] == r"\begin{align} z \end{align}"


def test_find_math_spans_nested_env_is_one_outer_span() -> None:
    text = r"\begin{align} x \begin{cases} y \end{cases} z \end{align}"

    assert find_math_spans(text) == [(0, len(text))]


def test_find_math_spans_escaped_dollar_is_not_math() -> None:
    text = r"Price \$5 and $x$"
    spans = find_math_spans(text)

    assert len(spans) == 1
    assert text[spans[0][0] : spans[0][1]] == "$x$"


def test_is_inside_math_uses_half_open_spans() -> None:
    spans = [(5, 10)]

    assert not is_inside_math(5, spans)
    assert is_inside_math(6, spans)
    assert not is_inside_math(10, spans)


def test_subsplit_never_corrupts_math() -> None:
    paragraphs = [
        "Paragraph zero " + "word " * 30,
        "Paragraph one " + "word " * 30,
        "Paragraph two has inline $x$ and " + "word " * 20,
        "Paragraph three " + "word " * 30,
        "Paragraph four " + "word " * 30,
        "Paragraph five " + "word " * 30,
        "Paragraph six " + "word " * 30,
        "Paragraph seven " + "word " * 30,
    ]
    math_blocks = [rf"\[ E_{i} = \frac{{1}}{{ {i + 1} }} \]" for i in range(6)]
    align_block = r"\begin{align} a &= b \\ c &= d \end{align}"

    body_parts: list[str] = []
    for i in range(6):
        body_parts.append(paragraphs[i])
        body_parts.append(math_blocks[i])
    body_parts.append(align_block)
    body_parts.append(paragraphs[6])
    body_parts.append(paragraphs[7])

    body = "\n\n".join(body_parts)
    section = _section(body, title="Long", page=2, path=("Doc", "Long"))
    config = _make_config(max_chunk_chars=220, min_chunk_chars=40)

    chunks = chunk_document(_make_document([section]), config)

    assert len(chunks) > 2
    assert "".join(chunk.text for chunk in chunks) == body

    for chunk in chunks:
        text = chunk.text
        assert text.count(r"\[") == text.count(r"\]")
        assert text.count(r"\begin{align}") == text.count(r"\end{align}")
        assert _unescaped_dollar_count(text) % 2 == 0

    left = re.escape(r"\[")
    right = re.escape(r"\]")
    for block in re.findall(f"{left}.*?{right}", body, re.DOTALL):
        containing_chunks = [index for index, chunk in enumerate(chunks) if block in chunk.text]
        assert len(containing_chunks) == 1, f"math block split or duplicated: {block}"


def test_single_equation_longer_than_max_is_kept_whole() -> None:
    body = r"\[" + "x" * 300 + r"\]"
    section = _section(body, title="Big Math", page=3, path=("Doc",))

    chunks = chunk_document(
        _make_document([section]),
        _make_config(max_chunk_chars=100, min_chunk_chars=20),
    )

    assert len(chunks) == 1
    assert chunks[0].text == body
    assert chunks[0].part == 1
    assert chunks[0].n_parts == 1


def test_subsplit_parts_share_section_metadata() -> None:
    body = "This is a sentence with enough words to drive a split. " * 25
    section = _section(
        body,
        title="Metadada",
        level=3,
        page=7,
        path=("Chapter", "Metadada"),
    )

    chunks = chunk_document(
        _make_document([section]),
        _make_config(max_chunk_chars=120, min_chunk_chars=20),
    )

    assert len(chunks) > 1
    n_parts = chunks[0].n_parts
    assert n_parts == len(chunks)
    assert [chunk.part for chunk in chunks] == list(range(1, n_parts + 1))

    for chunk in chunks:
        assert chunk.section_path == "Chapter > Metadada"
        assert chunk.section_title == "Metadada"
        assert chunk.level == 3
        assert chunk.page == 7
        assert chunk.n_parts == n_parts


def test_chunk_ids_are_unique_and_stable() -> None:
    body = "This is a sentence with enough words to drive a split. " * 25
    section = _section(body, title="ID Test", page=8, path=("Doc", "ID Test"))

    chunks = chunk_document(
        _make_document([section]),
        _make_config(max_chunk_chars=120, min_chunk_chars=20),
    )

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.chunk_id == f"{chunk.content_hash}-{chunk.part}"


def test_tiny_sections_dropped_or_kept_based_on_math() -> None:
    tiny_no_math = "tiny"
    tiny_math = r"$a$"
    sections = [
        _section(tiny_no_math, title="No math", path=("Root",)),
        _section(tiny_math, title="Math", page=4, path=("Root",)),
    ]

    chunks = chunk_document(_make_document(sections), _make_config(min_chunk_chars=80))

    assert len(chunks) == 1
    assert chunks[0].section_title == "Math"
    assert chunks[0].text == tiny_math


def test_chunk_embedding_text_prepends_section_path() -> None:
    chunk = Chunk(
        text="body text",
        source_path=Path("/tmp/test.tex"),
        filename="test.tex",
        section_path="A > B",
        section_title="B",
        level=2,
        page=None,
        math_fidelity="exact",
        part=1,
        n_parts=1,
        content_hash="abc123",
        chunk_id="abc123-1",
    )

    assert chunk_embedding_text(chunk) == "A > B\n\nbody text"

    empty_path_chunk = Chunk(
        text="body text",
        source_path=Path("/tmp/test.tex"),
        filename="test.tex",
        section_path="",
        section_title="",
        level=2,
        page=None,
        math_fidelity="exact",
        part=1,
        n_parts=1,
        content_hash="abc123",
        chunk_id="abc123-1",
    )
    assert chunk_embedding_text(empty_path_chunk) == "body text"


def test_chunk_documents_collects_all_documents() -> None:
    doc1 = _make_document([_section("word " * 30, title="One", path=("Doc", "One"))])
    doc2 = _make_document([_section("word " * 30, title="Two", path=("Doc", "Two"))])

    chunks = chunk_documents([doc1, doc2], _make_config())

    assert len(chunks) == 2
    assert chunks[0].section_title == "One"
    assert chunks[1].section_title == "Two"
