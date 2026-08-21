from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from physics_rag.chunking import chunk_document, chunk_embedding_text
from physics_rag.config import Config
from physics_rag.embeddings import Embedder
from physics_rag.parsers.base import ParsedDocument, Parser
from physics_rag.parsers.pdf import PdfParser
from physics_rag.parsers.tex import TexParser
from physics_rag.store import VectorStore

_LATEX_ARTEFACT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".aux",
        ".log",
        ".out",
        ".toc",
        ".synctex",
        ".fls",
        ".fdb_latexmk",
        ".bbl",
        ".blg",
        ".nav",
        ".snm",
        ".vrb",
    }
)


@dataclass(frozen=True, slots=True)
class IngestStats:
    files_seen: int = 0
    files_parsed: int = 0
    files_skipped_unchanged: int = 0
    files_skipped_duplicate: int = 0
    files_failed: int = 0
    chunks_added: int = 0


def discover_files(root: Path, config: Config) -> list[Path]:
    """Walk *root*, keeping only ingestable files and pruning excluded directories."""
    include_suffixes = {suffix.lower() for suffix in config.include_suffixes}
    exclude_dirs = set(config.exclude_dirs)
    found: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames if name not in exclude_dirs and not name.startswith(".")
        ]
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            suffix = path.suffix.lower()
            if suffix in _LATEX_ARTEFACT_SUFFIXES:
                continue
            if suffix not in include_suffixes:
                continue
            found.append(path)

    return sorted(found)


def dedupe_sources(paths: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    """Drop compiled PDFs that sit beside their .tex source.

    The .tex is preferred because its math is verbatim, whereas pdftotext
    flattens sub/superscripts. Indexing both would put near-identical text in
    the store and inflate retrieval hit-rate.
    """
    tex_siblings = {(path.parent, path.stem) for path in paths if path.suffix.lower() == ".tex"}

    kept: list[Path] = []
    dropped: list[Path] = []

    for path in paths:
        if path.suffix.lower() == ".pdf" and (path.parent, path.stem) in tex_siblings:
            dropped.append(path)
        else:
            kept.append(path)

    return kept, dropped


class IngestState:
    """Per-file (size, mtime) record enabling incremental, resumable ingest."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.files: dict[str, dict[str, object]] = {}
        self._load()

    @classmethod
    def load(cls, state_path: Path) -> IngestState:
        return cls(state_path)

    def _load(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError, UnicodeDecodeError:
            return

        if not isinstance(data, dict):
            return

        raw_files = data.get("files")
        if not isinstance(raw_files, dict):
            return

        files: dict[str, dict[str, object]] = {}
        for path_str, entry in raw_files.items():
            if not isinstance(path_str, str) or not isinstance(entry, dict):
                continue

            size = entry.get("size")
            mtime = entry.get("mtime")
            chunk_ids = entry.get("chunk_ids")

            if not isinstance(size, int) or not isinstance(mtime, int | float):
                continue
            if not isinstance(chunk_ids, list) or not all(
                isinstance(chunk_id, str) for chunk_id in chunk_ids
            ):
                continue

            files[path_str] = {
                "size": size,
                "mtime": float(mtime),
                "chunk_ids": chunk_ids,
            }

        self.files = files

    def is_unchanged(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except OSError:
            return False

        entry = self.files.get(str(path))
        if entry is None:
            return False

        return entry["size"] == stat.st_size and entry["mtime"] == stat.st_mtime

    def record(self, path: Path, chunk_ids: Sequence[str]) -> None:
        stat = path.stat()
        self.files[str(path)] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "chunk_ids": list(chunk_ids),
        }

    def forget(self, path: Path) -> None:
        self.files.pop(str(path), None)

    def save(self) -> None:
        """Write atomically so an interrupted ingest cannot corrupt the state."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(
            prefix=".ingest-state-",
            suffix=".json",
            dir=self.state_path.parent,
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
                json.dump({"files": self.files}, file_obj, indent=2)
            os.replace(tmp_path, self.state_path)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def parse_file(path: Path, parsers: Sequence[Parser]) -> ParsedDocument | None:
    for parser in parsers:
        if parser.can_parse(path):
            return parser.parse(path)
    return None


def _save_state_if_needed(state: IngestState | None, processed: int) -> None:
    if state is not None and processed % 25 == 0:
        state.save()


def ingest(
    root: Path,
    *,
    config: Config,
    embedder: Embedder,
    store: VectorStore,
    parsers: Sequence[Parser] | None = None,
    state: IngestState | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
    embed_batch_size: int = 64,
) -> IngestStats:
    """Parse, chunk, embed and store every ingestable file under *root*.

    One malformed file must never abort a multi-thousand-file run, so parse and
    chunk failures are counted and skipped rather than raised.
    """
    if embed_batch_size <= 0:
        raise ValueError("embed_batch_size must be positive")

    paths = discover_files(root, config)
    kept_paths, dropped_paths = dedupe_sources(paths)
    active_parsers = list(parsers) if parsers is not None else [TexParser(), PdfParser()]

    stats = IngestStats(
        files_seen=len(paths),
        files_skipped_duplicate=len(dropped_paths),
    )

    processed = 0
    for path in kept_paths:
        processed += 1

        if state is not None and not force and state.is_unchanged(path):
            stats = dataclasses.replace(
                stats,
                files_skipped_unchanged=stats.files_skipped_unchanged + 1,
            )
            if progress:
                progress(f"skipped unchanged: {path}")
            _save_state_if_needed(state, processed)
            continue

        try:
            doc = parse_file(path, active_parsers)
            if doc is None:
                if progress:
                    progress(f"skipped (no parser): {path}")
                _save_state_if_needed(state, processed)
                continue

            chunks = chunk_document(doc, config)
        except Exception as exc:
            stats = dataclasses.replace(stats, files_failed=stats.files_failed + 1)
            if progress:
                progress(f"failed: {path}: {exc}")
            _save_state_if_needed(state, processed)
            continue

        stats = dataclasses.replace(stats, files_parsed=stats.files_parsed + 1)

        # Clear any chunks from a previous version of this file first.
        store.delete_by_source(str(path))

        if not chunks:
            if state is not None:
                state.record(path, [])
            if progress:
                progress(f"no chunks: {path}")
            _save_state_if_needed(state, processed)
            continue

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        total_added = 0

        for start in range(0, len(chunks), embed_batch_size):
            batch = chunks[start : start + embed_batch_size]
            texts = [chunk_embedding_text(chunk) for chunk in batch]
            vectors = embedder.embed_documents(texts)
            store.add(batch, vectors)
            total_added += len(batch)

        stats = dataclasses.replace(stats, chunks_added=stats.chunks_added + total_added)

        if state is not None:
            state.record(path, chunk_ids)

        if progress:
            progress(f"ingested: {path} ({len(chunks)} chunks)")

        _save_state_if_needed(state, processed)

    if state is not None:
        state.save()

    return stats
