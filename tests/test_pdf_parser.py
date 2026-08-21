from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from physics_rag.normalize import BIDI_CONTROLS
from physics_rag.parsers.base import MathFidelity
from physics_rag.parsers.pdf import PDFTOTEXT, PdfExtractionError, PdfParser

FIXTURES = Path(__file__).parent / "fixtures"

requires_pdftotext = pytest.mark.skipif(
    shutil.which(PDFTOTEXT) is None,
    reason="pdftotext binary is not available",
)


@requires_pdftotext
def test_parse_hebrew_notes() -> None:
    doc = PdfParser().parse(FIXTURES / "hebrew_notes.pdf")

    assert doc.math_fidelity is MathFidelity.DEGRADED
    assert len(doc.sections) == 2
    assert [section.page for section in doc.sections] == [1, 2]

    # Each page is attributed to the deepest bookmark at or before it, and carries
    # the full ancestor chain so a citation can render "section > subsection".
    assert [section.title for section in doc.sections] == ["טמפרטורה", "פונקציית החלוקה"]
    assert doc.sections[0].path == ("אנטרופיה של מוצק איינשטיין", "טמפרטורה")
    assert doc.sections[0].section_path == "אנטרופיה של מוצק איינשטיין > טמפרטורה"
    assert doc.sections[1].path == ("פונקציית החלוקה",)

    for section in doc.sections:
        assert not any(ch in BIDI_CONTROLS for ch in section.body)


@requires_pdftotext
def test_english_pdf_with_outline_has_pages() -> None:
    doc = PdfParser().parse(FIXTURES / "hebrew_summary.pdf")

    assert [section.page for section in doc.sections] == [1, 2]
    assert "Two-Week Summary" in doc.sections[0].body


def test_pdf_extraction_error_for_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "physics_rag.parsers.pdf.PDFTOTEXT",
        "definitely-not-a-real-pdftotext-binary",
    )

    parser = PdfParser()

    with pytest.raises(PdfExtractionError):
        parser.parse(FIXTURES / "hebrew_notes.pdf")
