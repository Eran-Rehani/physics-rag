from __future__ import annotations

from physics_rag.normalize import BIDI_CONTROLS, content_hash, normalize_text, strip_bidi


def test_strip_bidi_removes_controls() -> None:
    text = "hello\u202b\u202a world\u200f"
    cleaned = strip_bidi(text)

    assert cleaned == "hello world"
    assert not any(ch in BIDI_CONTROLS for ch in cleaned)


def test_normalize_is_idempotent() -> None:
    sample = "  שלום עולם\r\n\r\n\r\n  math: $a_b$  \n\n\n\n  end \r"
    once = normalize_text(sample)

    assert normalize_text(once) == once


def test_normalize_nfc() -> None:
    assert normalize_text("e\u0301") == "\u00e9"


def test_collapse_blank_lines() -> None:
    assert normalize_text("a\n\n\n\nb") == "a\n\nb"


def test_content_hash_stable_and_equal_for_bidi_whitespace() -> None:
    a = "שלום עולם"
    b = "\u202bשלום\u202c עולם\n"

    assert content_hash(a) == content_hash(b)
    assert len(content_hash(a)) == 16
