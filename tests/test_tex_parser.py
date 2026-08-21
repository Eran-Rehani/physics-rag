from __future__ import annotations

from pathlib import Path

from physics_rag.parsers.base import MathFidelity
from physics_rag.parsers.tex import TexParser

FIXTURES = Path(__file__).parent / "fixtures"


def test_can_parse() -> None:
    parser = TexParser()

    assert parser.can_parse(Path("x.tex"))
    assert not parser.can_parse(Path("x.pdf"))


def test_parse_real_fixture_math_and_sections() -> None:
    parser = TexParser()
    doc = parser.parse(FIXTURES / "hebrew_lecture.tex")

    assert doc.math_fidelity is MathFidelity.EXACT

    all_bodies = "\n".join(section.body for section in doc.sections)
    assert r"\documentclass" not in all_bodies
    assert r"\usepackage" not in all_bodies

    # A heading whose body is empty (here the \section, immediately followed by a
    # \subsection) yields no chunk of its own - there is nothing to embed. Its title
    # must still survive in the descendants' path so citations resolve to it.
    titles = [section.title for section in doc.sections]
    assert titles == ["תקציר השבוע הקודם", "מוצק אינשטיין:"]

    section_title = "שבוע 2 - הרצאה 3 28 אוקטובר 2025"
    assert all(section.path[0] == section_title for section in doc.sections)

    subsubsection = next(section for section in doc.sections if section.title == "מוצק אינשטיין:")
    assert subsubsection.path == (
        section_title,
        "תקציר השבוע הקודם",
        "מוצק אינשטיין:",
    )
    assert subsubsection.level == 3
    assert subsubsection.section_path == " > ".join(subsubsection.path)

    # Display math must survive byte-identically; this is a hard project requirement.
    assert r"S=k_{B}\ln\left(\Omega\right)" in all_bodies
    assert r"\Omega\left(N+1,q\right)=\frac{\left(N+q\right)!}{q!N!}" in all_bodies


def test_comments_stripped_and_percent_literal_survives(tmp_path: Path) -> None:
    tex = tmp_path / "example.tex"
    tex.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "% comment\n"
        "100\\% efficiency\n"
        "\\end{document}\n",
        encoding="utf-8",
    )

    doc = TexParser().parse(tex)
    body = doc.sections[0].body

    assert "comment" not in body
    assert r"100\% efficiency" in body


def test_formatting_wrapper_unwrapped(tmp_path: Path) -> None:
    tex = tmp_path / "formatting.tex"
    tex.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\textbf{\\textcolor{red}{X}}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )

    doc = TexParser().parse(tex)

    assert doc.sections[0].body == "X"
