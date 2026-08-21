from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    corpus_root: Path = field(default_factory=lambda: Path.home() / "GoogleDrive")
    staging_dir: Path = field(default_factory=lambda: Path.home() / ".cache/physics-rag/corpus")
    chroma_dir: Path = field(default_factory=lambda: Path.home() / ".cache/physics-rag/chroma")
    collection_name: str = "physics"
    embedding_model: str = "intfloat/multilingual-e5-small"
    max_chunk_chars: int = 1800
    min_chunk_chars: int = 80
    chunk_overlap_chars: int = 0
    top_k: int = 6
    abstain_threshold: float = 0.35
    llama_server_url: str = "http://127.0.0.1:8080"
    generation_timeout: float = 300.0
    exclude_dirs: tuple[str, ...] = ("Docs",)
    include_suffixes: tuple[str, ...] = (".tex", ".pdf")


def default_config() -> Config:
    return Config()


DEFAULT_CONFIG = default_config()

_PATH_KEYS = frozenset({"corpus_root", "staging_dir", "chroma_dir"})
_TUPLE_KEYS = frozenset({"exclude_dirs", "include_suffixes"})


def from_toml(path: str | Path) -> Config:
    """Load config from a TOML file's ``[physics_rag]`` table, overriding defaults."""
    path = Path(path).expanduser()
    with path.open("rb") as f:
        data = tomllib.load(f)

    table = data.get("physics_rag")
    if table is None:
        return default_config()
    if not isinstance(table, dict):
        raise ValueError("The [physics_rag] TOML table must be a table")

    known = {f.name for f in fields(Config)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")

    overrides: dict[str, object] = {}
    for key, value in table.items():
        if key in _PATH_KEYS:
            value = Path(str(value)).expanduser()
        elif key in _TUPLE_KEYS:
            value = tuple(value)
        overrides[key] = value

    return Config(**overrides)
