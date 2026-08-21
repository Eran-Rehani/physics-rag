from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from physics_rag.config import Config
from physics_rag.generation import Generator
from physics_rag.retrieval import Retriever, dedupe_results, format_citation
from physics_rag.store import SearchResult

ABSTAIN_MESSAGE = "not found in corpus"

SYSTEM_PROMPT = (
    "You answer physics questions strictly from the sources provided to you.\n\n"
    "Rules:\n"
    "1. Answer ONLY from the provided sources. If the sources do not contain the answer, "
    f"reply exactly: {ABSTAIN_MESSAGE}\n"
    "2. Cite every claim inline by copying the full bracket label given after 'cite as', "
    "for example [Lec_03.tex, Entropy]. Never cite a bare source number such as [3], and "
    "never invent a filename, section or page.\n"
    "3. Reproduce LaTeX equations ONLY from sources marked (math: exact). Sources marked "
    "(math: degraded) had their equations flattened by PDF text extraction, so their formulas "
    "are unreliable: describe those in words, or quote them with an explicit caveat, but never "
    "present them as exact LaTeX.\n"
    "4. Answer in English, even when the sources are written in Hebrew.\n"
)

# Labels contain Hebrew and " > " separators; keep it non-greedy and single-line.
_LABEL_RE = re.compile(r"\[[^\[\]\n]+\]")
# Models cite the block either as "[3]" or as "[SOURCE 3]"; catch both.
_NUMERIC_REF_RE = re.compile(r"\[(?:source\s*)?(\d{1,2})\]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    text: str
    abstained: bool
    confidence: float
    citations: list[str]
    results: list[SearchResult]


def build_context(results: Sequence[SearchResult]) -> str:
    """Render retrieved chunks as numbered, citable source blocks."""
    blocks = [
        f"SOURCE {index} -- cite as {format_citation(result)} "
        f"(math: {result.math_fidelity})\n{result.text}"
        for index, result in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks)


def resolve_numeric_citations(text: str, results: Sequence[SearchResult]) -> str:
    """Rewrite bare source numbers like ``[3]`` into their real citation labels.

    Small models sometimes cite the source number instead of the label. The
    requirement is that every claim carries [filename, section/page], so the
    number is resolved back rather than left as an unusable reference.
    """

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 1 <= index <= len(results):
            return format_citation(results[index - 1])
        return match.group(0)

    return _NUMERIC_REF_RE.sub(replace, text)


def build_prompt(question: str, results: Sequence[SearchResult]) -> str:
    return f"Sources:\n\n{build_context(results)}\n\nQuestion: {question}"


def extract_cited_labels(text: str) -> list[str]:
    """Every bracket label in *text*, in order, de-duplicated, brackets included."""
    return list(dict.fromkeys(_LABEL_RE.findall(text)))


class AnswerService:
    """Retrieve, then either abstain or generate a cited answer."""

    def __init__(self, retriever: Retriever, generator: Generator, config: Config) -> None:
        self._retriever = retriever
        self._generator = generator
        self._config = config

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def generator(self) -> Generator:
        return self._generator

    def with_config(self, config: Config) -> AnswerService:
        """A service sharing this one's retriever and generator, under a new config.

        Used by the UI to vary the abstain threshold without reloading the
        embedding model or reconnecting to llama-server.
        """
        return AnswerService(self._retriever, self._generator, config)

    def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        retrieval = self._retriever.retrieve(question, top_k=top_k)
        results = dedupe_results(retrieval.results)

        # The service owns the abstain decision rather than deferring to the
        # retriever's own config, so a caller can vary the threshold (the UI
        # slider, the eval sweep) without rebuilding the retriever.
        abstain = retrieval.confidence < self._config.abstain_threshold

        # Never spend a generation on a low-confidence retrieval.
        if abstain or not results:
            return Answer(
                question=question,
                text=ABSTAIN_MESSAGE,
                abstained=True,
                confidence=retrieval.confidence,
                citations=[],
                results=results,
            )

        text = self._generator.generate(build_prompt(question, results), system=SYSTEM_PROMPT)

        text = resolve_numeric_citations(text, results)

        abstained = text.strip().casefold().startswith(ABSTAIN_MESSAGE)
        if abstained:
            text = ABSTAIN_MESSAGE

        return Answer(
            question=question,
            text=text,
            abstained=abstained,
            confidence=retrieval.confidence,
            citations=[] if abstained else [format_citation(result) for result in results],
            results=results,
        )
