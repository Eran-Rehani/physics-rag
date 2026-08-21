from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from physics_rag.normalize import normalize_text
from physics_rag.parsers.base import MathFidelity, ParsedDocument, Section

PDFTOTEXT = "pdftotext"


class PdfExtractionError(RuntimeError):
    """Raised when pdftotext cannot extract text from a PDF."""


class PdfParser:
    def __init__(self, timeout: float = 120) -> None:
        self.timeout = timeout

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> ParsedDocument:
        raw_text = self._extract_text(path)
        pages = raw_text.split("\f")
        outline = self._get_outline(path)

        sections: list[Section] = []
        outline_idx = 0
        stack: list[tuple[int, str]] = []

        for page_no, page_text in enumerate(pages, start=1):
            page_index = page_no - 1

            while outline_idx < len(outline) and outline[outline_idx][0] <= page_index:
                _, raw_title, level = outline[outline_idx]
                title = normalize_text(raw_title)

                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                outline_idx += 1

            if stack:
                title = stack[-1][1]
                level = stack[-1][0]
                section_path = tuple(t for _, t in stack)
            else:
                title = ""
                level = 0
                section_path = ()

            body = normalize_text(page_text)
            if not body:
                continue

            sections.append(
                Section(
                    title=title,
                    level=level,
                    body=body,
                    page=page_no,
                    path=section_path,
                )
            )

        return ParsedDocument(
            source_path=path,
            sections=sections,
            math_fidelity=MathFidelity.DEGRADED,
        )

    def _extract_text(self, path: Path) -> str:
        cmd = [PDFTOTEXT, "-layout", "-enc", "UTF-8", str(path), "-"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise PdfExtractionError(f"pdftotext binary not found: {PDFTOTEXT}") from exc
        except subprocess.TimeoutExpired as exc:
            raise PdfExtractionError(
                f"pdftotext timed out after {self.timeout}s for {path}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = exc.stderr or exc.stdout
            raise PdfExtractionError(
                f"pdftotext failed for {path} with exit code {exc.returncode}: {details}"
            ) from exc

        return result.stdout

    def _get_outline(self, path: Path) -> list[tuple[int, str, int]]:
        try:
            reader = PdfReader(str(path))
            items = reader.outline
        except Exception:
            return []

        result: list[tuple[int, str, int]] = []

        # pypdf represents nested bookmarks as nested lists inside reader.outline,
        # so recursing into list/tuple entries is sufficient. Note that
        # Destination.children is a *method*, not a sequence - do not walk it.
        def walk(entries: Any, level: int) -> None:
            if not entries:
                return

            for entry in entries:
                if isinstance(entry, list | tuple):
                    walk(entry, level + 1)
                    continue

                try:
                    title = entry.title
                    page = reader.get_destination_page_number(entry)
                    result.append((int(page), str(title), level))
                except Exception:
                    pass

        walk(items, 1)
        return sorted(result, key=lambda x: x[0])
