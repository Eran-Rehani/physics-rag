from __future__ import annotations

import re
from pathlib import Path

from physics_rag.normalize import normalize_text
from physics_rag.parsers.base import MathFidelity, ParsedDocument, Section

_LEVELS = {
    "chapter": 0,
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
}

_HEADING_RE = re.compile(
    r"\\(chapter|section|subsection|subsubsection|paragraph)\*?(?:\[[^\]]*\])?\{"
)

_MATH_ENV_BASES = {
    "displaymath",
    "equation",
    "align",
    "eqnarray",
    "gather",
    "multline",
    "math",
}

_FORMAT_COMMANDS = ("textcolor", "textbf", "textit", "emph", "text")


def strip_preamble(src: str) -> str:
    r"""Return the body after the first ``\begin{document}``, if present."""
    begin = src.find(r"\begin{document}")
    if begin == -1:
        return src

    body_start = begin + len(r"\begin{document}")
    body = src[body_start:]

    end = body.rfind(r"\end{document}")
    if end != -1:
        body = body[:end]

    return body


def _strip_comments(src: str) -> str:
    r"""Remove LaTeX comments while preserving ``\%``."""
    out: list[str] = []
    i = 0
    while i < len(src):
        if src[i] == "%" and (i == 0 or src[i - 1] != "\\"):
            newline = src.find("\n", i)
            if newline == -1:
                break
            out.append("\n")
            i = newline + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def _extract_braced(text: str, open_idx: int) -> tuple[str, int]:
    """Return content and index-after-closing-brace for ``{...}``."""
    if open_idx >= len(text) or text[open_idx] != "{":
        raise ValueError(f"Expected '{{' at index {open_idx}")

    depth = 1
    i = open_idx + 1
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            i += 2
            continue

        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
        i += 1

    raise ValueError("Unbalanced braces")


def _is_math_env(env_name: str) -> bool:
    return env_name.rstrip("*") in _MATH_ENV_BASES


def _clean_chunk(text: str) -> str:
    """Remove lightweight LaTeX formatting commands while preserving math verbatim."""
    out: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        char = text[i]

        if char == "$":
            if i + 1 < n and text[i + 1] == "$":
                end = text.find("$$", i + 2)
                if end == -1:
                    out.append(text[i:])
                    break
                out.append(text[i : end + 2])
                i = end + 2
                continue

            end = text.find("$", i + 1)
            if end == -1:
                out.append(text[i:])
                break
            out.append(text[i : end + 1])
            i = end + 1
            continue

        if text.startswith(r"\(", i):
            end = text.find(r"\)", i + 2)
            if end == -1:
                out.append(text[i:])
                break
            out.append(text[i : end + 2])
            i = end + 2
            continue

        if text.startswith(r"\[", i):
            end = text.find(r"\]", i + 2)
            if end == -1:
                out.append(text[i:])
                break
            out.append(text[i : end + 2])
            i = end + 2
            continue

        if text.startswith(r"\begin{", i):
            env_start = i + 6
            try:
                env_name, after_env = _extract_braced(text, env_start)
            except ValueError:
                out.append(char)
                i += 1
                continue

            if _is_math_env(env_name):
                end_tag = r"\end{" + env_name + r"}"
                end = text.find(end_tag, after_env)
                if end == -1:
                    out.append(text[i:])
                    break
                out.append(text[i : end + len(end_tag)])
                i = end + len(end_tag)
                continue

        if text.startswith(r"\label{", i):
            try:
                _, after_label = _extract_braced(text, i + 6)
                i = after_label
                continue
            except ValueError:
                out.append(char)
                i += 1
                continue

        handled = False
        for cmd in _FORMAT_COMMANDS:
            prefix = "\\" + cmd + "{"
            if text.startswith(prefix, i):
                open_idx = i + len(cmd) + 1
                try:
                    if cmd == "textcolor":
                        _, after_color = _extract_braced(text, open_idx)
                        if after_color < n and text[after_color] == "{":
                            inner, after_inner = _extract_braced(text, after_color)
                            out.append(_clean_chunk(inner))
                            i = after_inner
                        else:
                            out.append(char)
                            i += 1
                    else:
                        inner, after_inner = _extract_braced(text, open_idx)
                        out.append(_clean_chunk(inner))
                        i = after_inner
                    handled = True
                except ValueError:
                    out.append(char)
                    i += 1
                    handled = True
                break

        if handled:
            continue

        out.append(char)
        i += 1

    return "".join(out)


def _clean_text(raw: str) -> str:
    return normalize_text(_clean_chunk(raw))


def _split_sections(body: str) -> list[tuple[str, int, tuple[str, ...], str]]:
    sections: list[tuple[str, int, tuple[str, ...], str]] = []

    pos = 0
    current_title = ""
    current_level = 0
    current_path: tuple[str, ...] = ()
    stack: list[tuple[int, str]] = []

    for match in _HEADING_RE.finditer(body):
        raw_body = body[pos : match.start()]
        sections.append((current_title, current_level, current_path, raw_body))

        level = _LEVELS[match.group(1)]
        title_raw, after_close = _extract_braced(body, match.end() - 1)

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title_raw))

        current_title = title_raw
        current_level = level
        current_path = tuple(t for _, t in stack)
        pos = after_close

    sections.append((current_title, current_level, current_path, body[pos:]))
    return sections


class TexParser:
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".tex"

    def parse(self, path: Path) -> ParsedDocument:
        src = path.read_text(encoding="utf-8")
        body = _strip_comments(strip_preamble(src))

        sections: list[Section] = []
        for raw_title, level, section_path, raw_body in _split_sections(body):
            title = _clean_text(raw_title)
            clean_body = _clean_text(raw_body)

            if not clean_body:
                continue

            sections.append(
                Section(
                    title=title,
                    level=level,
                    body=clean_body,
                    page=None,
                    path=section_path,
                )
            )

        return ParsedDocument(
            source_path=path,
            sections=sections,
            math_fidelity=MathFidelity.EXACT,
        )
