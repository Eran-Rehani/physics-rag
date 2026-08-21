from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class MathFidelity(StrEnum):
    EXACT = "exact"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    level: int
    body: str
    page: int | None = None
    path: tuple[str, ...] = ()

    @property
    def section_path(self) -> str:
        return " > ".join(self.path)


@dataclass(slots=True)
class ParsedDocument:
    source_path: Path
    sections: list[Section]
    math_fidelity: MathFidelity
    doc_title: str | None = None


class Parser(Protocol):
    def can_parse(self, path: Path) -> bool: ...

    def parse(self, path: Path) -> ParsedDocument: ...
