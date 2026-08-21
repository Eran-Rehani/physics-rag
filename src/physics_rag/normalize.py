from __future__ import annotations

import hashlib
import re
import unicodedata

BIDI_CONTROLS = frozenset(
    {
        "‎",
        "‏",
        *[chr(codepoint) for codepoint in range(0x202A, 0x202F)],
        *[chr(codepoint) for codepoint in range(0x2066, 0x206A)],
    }
)


def strip_bidi(text: str) -> str:
    """Remove Unicode bidi control characters from *text*."""
    if not text:
        return ""
    return "".join(ch for ch in text if ch not in BIDI_CONTROLS)


def normalize_text(text: str) -> str:
    """Return a stable, idempotent normalization of *text*.

    NFC-normalizes, strips bidi controls, unifies line endings, collapses runs of
    spaces/tabs, and collapses consecutive blank lines to a single blank line so
    that the same content reached via .tex and via PDF extraction hashes equal.
    """
    text = unicodedata.normalize("NFC", text)
    text = strip_bidi(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    lines = [line.rstrip() for line in text.split("\n")]

    collapsed: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            if blank_count < 1:
                collapsed.append(line)
            blank_count += 1
        else:
            blank_count = 0
            collapsed.append(line)

    return "\n".join(collapsed).strip()


def content_hash(text: str) -> str:
    """Return the first 16 hex characters of the SHA-256 hash of normalized text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]
