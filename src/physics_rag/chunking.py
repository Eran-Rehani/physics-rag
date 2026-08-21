from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from physics_rag.config import Config
from physics_rag.normalize import content_hash
from physics_rag.parsers.base import ParsedDocument, Section

_MATH_ENVIRONMENTS = frozenset(
    {
        "equation",
        "align",
        "aligned",
        "gather",
        "gathered",
        "multline",
        "eqnarray",
        "displaymath",
        "math",
        "split",
        "cases",
        "array",
        "matrix",
        "pmatrix",
        "bmatrix",
        "vmatrix",
        "Vmatrix",
        "smallmatrix",
    }
)


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    source_path: Path
    filename: str
    section_path: str
    section_title: str
    level: int
    page: int | None
    math_fidelity: str
    part: int
    n_parts: int
    content_hash: str
    chunk_id: str


def _is_escaped(text: str, pos: int) -> bool:
    backslashes = 0
    i = pos - 1
    while i >= 0 and text[i] == "\\":
        backslashes += 1
        i -= 1
    return backslashes % 2 == 1


def _normalise_env_name(name: str) -> str | None:
    name = name.strip()
    if not name:
        return None
    if name.endswith("*"):
        name = name[:-1]
    return name if name in _MATH_ENVIRONMENTS else None


def _parse_math_env_name(text: str, start: int) -> str | None:
    if not text.startswith("\\begin{", start):
        return None
    name_start = start + len("\\begin{")
    close = text.find("}", name_start)
    if close == -1:
        return None
    return _normalise_env_name(text[name_start:close])


def _find_env_end(text: str, start: int, env_name: str) -> int | None:
    n = len(text)
    close_brace = text.find("}", start)
    if close_brace == -1:
        return None

    stack = [env_name]
    i = close_brace + 1

    while i < n:
        if text.startswith("\\begin{", i) and not _is_escaped(text, i):
            nested_name = _parse_math_env_name(text, i)
            if nested_name is not None:
                stack.append(nested_name)
                nested_close = text.find("}", i)
                i = nested_close + 1 if nested_close != -1 else i + len("\\begin{")
                continue

        if text.startswith("\\end{", i) and not _is_escaped(text, i):
            name_start = i + len("\\end{")
            close = text.find("}", name_start)
            if close != -1:
                name = _normalise_env_name(text[name_start:close])
                if name is not None and stack and stack[-1] == name:
                    stack.pop()
                    if not stack:
                        return close + 1
                i = close + 1
                continue

        i += 1

    return None


def find_math_spans(text: str) -> list[tuple[int, int]]:
    """Return half-open [start, end) spans covering every math region in *text*."""
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0

    while i < n:
        if text.startswith("\\begin{", i) and not _is_escaped(text, i):
            env_name = _parse_math_env_name(text, i)
            if env_name is not None:
                end = _find_env_end(text, i, env_name)
                if end is None:
                    spans.append((i, n))
                    break
                spans.append((i, end))
                i = end
                continue

        if (text.startswith("\\[", i) or text.startswith("\\(", i)) and not _is_escaped(text, i):
            opener = text[i : i + 2]
            closer = "\\]" if opener == "\\[" else "\\)"
            j = i + 2
            while j < n:
                if text.startswith(closer, j) and not _is_escaped(text, j):
                    spans.append((i, j + 2))
                    i = j + 2
                    break
                j += 1
            else:
                spans.append((i, n))
                break
            continue

        if text.startswith("$$", i) and not _is_escaped(text, i):
            j = i + 2
            while j < n:
                if text.startswith("$$", j) and not _is_escaped(text, j):
                    spans.append((i, j + 2))
                    i = j + 2
                    break
                j += 1
            else:
                spans.append((i, n))
                break
            continue

        if text[i] == "$" and not _is_escaped(text, i):
            j = i + 1
            while j < n:
                if text[j] == "$" and not _is_escaped(text, j):
                    spans.append((i, j + 1))
                    i = j + 1
                    break
                j += 1
            else:
                spans.append((i, n))
                break
            continue

        i += 1

    return spans


def is_inside_math(pos: int, spans: list[tuple[int, int]]) -> bool:
    """True if splitting at *pos* would cut a math region in half.

    Boundaries are safe: a split exactly at a span's start leaves the equation
    whole at the head of the next chunk, and a split at its end leaves it whole
    at the tail of the previous one. Only strictly interior positions corrupt math.
    """
    return any(start < pos < end for start, end in spans)


def _candidate_positions(text: str, spans: list[tuple[int, int]]) -> list[int]:
    positions: set[int] = set()

    for match in re.finditer(r"[ \t]*(?:\n[ \t]*)+", text):
        positions.add(match.end())

    for match in re.finditer(r"[.!?] +", text):
        positions.add(match.end())

    for _, end in spans:
        if end < len(text):
            positions.add(end)

    return sorted(positions)


def _choose_split(
    text: str,
    pos: int,
    max_chars: int,
    spans: list[tuple[int, int]],
    candidates: list[int],
) -> int:
    limit = pos + max_chars
    n = len(text)

    if limit >= n:
        return n

    idx = bisect_right(candidates, limit) - 1
    while idx >= 0:
        candidate = candidates[idx]
        if candidate <= pos:
            break
        if not is_inside_math(candidate, spans):
            return candidate
        idx -= 1

    if not is_inside_math(limit, spans):
        return limit

    idx = bisect_right(candidates, limit)
    while idx < len(candidates):
        candidate = candidates[idx]
        if candidate > limit and not is_inside_math(candidate, spans):
            return candidate
        idx += 1

    return n


def _subsplit_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    spans = find_math_spans(text)
    candidates = _candidate_positions(text, spans)

    parts: list[str] = []
    pos = 0
    n = len(text)

    while pos < n:
        if n - pos <= max_chars:
            parts.append(text[pos:])
            break

        split_pos = _choose_split(text, pos, max_chars, spans, candidates)

        if split_pos <= pos:
            split_pos = min(pos + max_chars, n)
            if split_pos <= pos:
                split_pos = pos + 1

        parts.append(text[pos:split_pos])
        pos = split_pos

    return parts


def _merge_short_parts(parts: list[str], min_chars: int) -> list[str]:
    merged: list[str] = []
    for part in parts:
        if len(part) < min_chars and merged:
            merged[-1] += part
        else:
            merged.append(part)
    return merged


def _contains_math(text: str) -> bool:
    return bool(find_math_spans(text))


def _build_chunk(
    doc: ParsedDocument,
    section: Section,
    text: str,
    part: int,
    n_parts: int,
) -> Chunk:
    digest = content_hash(text)
    return Chunk(
        text=text,
        source_path=doc.source_path,
        filename=doc.source_path.name,
        section_path=section.section_path,
        section_title=section.title,
        level=section.level,
        page=section.page,
        math_fidelity=str(doc.math_fidelity),
        part=part,
        n_parts=n_parts,
        content_hash=digest,
        chunk_id=f"{digest}-{part}",
    )


def chunk_document(doc: ParsedDocument, config: Config) -> list[Chunk]:
    """Split *doc* into chunks, using sections as the primary boundary."""
    chunks: list[Chunk] = []

    for section in doc.sections:
        body = section.body
        if body == "":
            continue

        if len(body) <= config.max_chunk_chars:
            if len(body) < config.min_chunk_chars and not _contains_math(body):
                continue
            parts = [body]
        else:
            parts = _subsplit_text(body, config.max_chunk_chars)
            parts = _merge_short_parts(parts, config.min_chunk_chars)

        n_parts = len(parts)
        for part_num, part_text in enumerate(parts, start=1):
            chunks.append(_build_chunk(doc, section, part_text, part_num, n_parts))

    return chunks


def chunk_documents(docs: Iterable[ParsedDocument], config: Config) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, config))
    return chunks


def chunk_embedding_text(chunk: Chunk) -> str:
    """Text actually embedded: the section path contributes retrieval signal."""
    if chunk.section_path:
        return f"{chunk.section_path}\n\n{chunk.text}"
    return chunk.text
