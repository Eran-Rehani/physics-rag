from __future__ import annotations

from physics_rag.store import SearchResult
from physics_rag.ui import format_sources_markdown


def _result(filename: str, section_path: str, fidelity: str = "exact") -> SearchResult:
    return SearchResult(
        chunk_id="c",
        text="t",
        score=0.912,
        filename=filename,
        section_path=section_path,
        page=None,
        math_fidelity=fidelity,
        source_path=f"/tmp/{filename}",
    )


def test_format_sources_markdown_renders_table() -> None:
    markdown = format_sources_markdown(
        [_result("a.tex", "One"), _result("b.pdf", "Two", "degraded")]
    )

    assert "| score | citation | math |" in markdown
    assert "[a.tex, One]" in markdown
    assert "degraded" in markdown


def test_format_sources_markdown_handles_empty() -> None:
    assert "No sources" in format_sources_markdown([])


def test_pipe_in_citation_is_escaped() -> None:
    markdown = format_sources_markdown([_result("a.tex", "One | Two")])

    assert r"One \| Two" in markdown
