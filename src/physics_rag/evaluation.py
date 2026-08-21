"""Scoring harness for the citation-grounded RAG eval set.

Scores retrieval hit-rate and citation correctness, and calibrates the
abstention threshold from measured separation between answerable questions and
deliberate negatives rather than leaving it a guess.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from physics_rag.answer import Answer, extract_cited_labels


class AnswerLike(Protocol):
    def ask(self, question: str, *, top_k: int | None = None) -> Answer: ...


@dataclass(frozen=True, slots=True)
class EvalItem:
    id: str
    question: str
    answerable: bool = True
    expected_files: tuple[str, ...] = ()
    expected_citations: tuple[tuple[str, str | None, int | None], ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ItemResult:
    item: EvalItem
    retrieved_files: tuple[str, ...]
    retrieval_hit: bool
    citation_correct: bool
    abstained: bool
    confidence: float
    answer_text: str
    correct_abstention: bool


@dataclass(frozen=True, slots=True)
class EvalReport:
    results: tuple[ItemResult, ...]
    n_answerable: int
    n_negative: int
    retrieval_hit_rate: float
    citation_accuracy: float
    abstention_precision: float
    abstention_recall: float
    abstention_f1: float
    mean_confidence_answerable: float
    mean_confidence_negative: float


def load_eval_set(path: Path) -> list[EvalItem]:
    """Parse the YAML eval set, failing loudly on anything malformed."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError("eval set must contain a top-level 'questions' list")
    raw_items = data["questions"]
    if not isinstance(raw_items, list):
        raise ValueError("eval set 'questions' must be a list")

    items: list[EvalItem] = []
    seen_ids: set[str] = set()

    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"item {idx}: must be a mapping")

        item_id = raw.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"item {idx}: missing or invalid id")
        if item_id in seen_ids:
            raise ValueError(f"duplicate id: {item_id}")
        seen_ids.add(item_id)

        question = raw.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"item {item_id}: missing question")

        answerable = raw.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"item {item_id}: answerable must be a boolean")

        raw_expected_files = raw.get("expected_files", [])
        if not isinstance(raw_expected_files, list):
            raise ValueError(f"item {item_id}: expected_files must be a list")
        expected_files = tuple(str(name) for name in raw_expected_files)

        raw_citations = raw.get("expected_citations", [])
        if not isinstance(raw_citations, list):
            raise ValueError(f"item {item_id}: expected_citations must be a list")

        citations: list[tuple[str, str | None, int | None]] = []
        for c_idx, citation in enumerate(raw_citations):
            if not isinstance(citation, dict):
                raise ValueError(f"item {item_id}: expected_citations[{c_idx}] must be a mapping")

            file = citation.get("file")
            if not isinstance(file, str) or not file.strip():
                raise ValueError(f"item {item_id}: expected_citations[{c_idx}] missing file")

            section = citation.get("section")
            if section is not None and not isinstance(section, str):
                raise ValueError(
                    f"item {item_id}: expected_citations[{c_idx}] section must be a string"
                )

            page = citation.get("page")
            if page is not None and (not isinstance(page, int) or isinstance(page, bool)):
                raise ValueError(f"item {item_id}: expected_citations[{c_idx}] invalid page")

            citations.append((file, section, page))

        if answerable and not expected_files:
            raise ValueError(f"item {item_id}: answerable item must have expected_files")
        if not answerable and expected_files:
            raise ValueError(f"item {item_id}: unanswerable item must not have expected_files")

        items.append(
            EvalItem(
                id=item_id,
                question=question,
                answerable=answerable,
                expected_files=expected_files,
                expected_citations=tuple(citations),
                notes=raw.get("notes", "") or "",
            )
        )

    return items


def normalise_filename(name: str) -> str:
    return name.replace("\\", "/").split("/")[-1].casefold()


def match_citation(label: str, expected: tuple[str, str | None, int | None]) -> bool:
    """Match a rendered citation label against an expected (file, section, page).

    Section paths contain " > " and may contain commas, so the label is split on
    the FIRST comma only.
    """
    expected_file, expected_section, expected_page = expected

    inner = label.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1].strip()

    filename, _, remainder = inner.partition(",")
    if normalise_filename(filename.strip()) != normalise_filename(expected_file):
        return False

    if expected_section is None and expected_page is None:
        return True

    remainder = remainder.strip()

    if expected_page is not None:
        return remainder == f"p. {expected_page}"

    if expected_section is not None:
        return expected_section.casefold() in remainder.casefold()

    return False


def _any_label_matches_files(labels: Sequence[str], expected_files: tuple[str, ...]) -> bool:
    return any(
        match_citation(label, (expected_file, None, None))
        for expected_file in expected_files
        for label in labels
    )


def score_item(item: EvalItem, answer: Answer) -> ItemResult:
    retrieved_files = tuple(result.filename for result in answer.results)

    if not item.answerable:
        return ItemResult(
            item=item,
            retrieved_files=retrieved_files,
            retrieval_hit=False,
            citation_correct=False,
            abstained=answer.abstained,
            confidence=answer.confidence,
            answer_text=answer.text,
            correct_abstention=answer.abstained,
        )

    expected = {normalise_filename(name) for name in item.expected_files}
    retrieval_hit = any(normalise_filename(name) in expected for name in retrieved_files)

    labels = extract_cited_labels(answer.text)
    if item.expected_citations:
        citation_correct = any(
            match_citation(label, expectation)
            for expectation in item.expected_citations
            for label in labels
        )
    else:
        citation_correct = _any_label_matches_files(labels, item.expected_files)

    return ItemResult(
        item=item,
        retrieved_files=retrieved_files,
        retrieval_hit=retrieval_hit,
        citation_correct=citation_correct,
        abstained=answer.abstained,
        confidence=answer.confidence,
        answer_text=answer.text,
        correct_abstention=not answer.abstained,
    )


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_eval(
    items: Sequence[EvalItem],
    service: AnswerLike,
    *,
    top_k: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> EvalReport:
    results: list[ItemResult] = []
    for item in items:
        if progress:
            progress(f"evaluating {item.id}")
        results.append(score_item(item, service.ask(item.question, top_k=top_k)))

    results_tuple = tuple(results)

    n_answerable = sum(1 for item in items if item.answerable)
    n_negative = len(items) - n_answerable

    hits = sum(1 for r in results_tuple if r.item.answerable and r.retrieval_hit)
    retrieval_hit_rate = hits / n_answerable if n_answerable else 0.0

    answered = [r for r in results_tuple if r.item.answerable and not r.abstained]
    citation_accuracy = (
        sum(1 for r in answered if r.citation_correct) / len(answered) if answered else 0.0
    )

    abstained = [r for r in results_tuple if r.abstained]
    abstention_precision = (
        sum(1 for r in abstained if not r.item.answerable) / len(abstained) if abstained else 0.0
    )

    negatives = [r for r in results_tuple if not r.item.answerable]
    abstention_recall = (
        sum(1 for r in negatives if r.abstained) / len(negatives) if negatives else 0.0
    )

    answerable_conf = [r.confidence for r in results_tuple if r.item.answerable]
    negative_conf = [r.confidence for r in results_tuple if not r.item.answerable]

    return EvalReport(
        results=results_tuple,
        n_answerable=n_answerable,
        n_negative=n_negative,
        retrieval_hit_rate=retrieval_hit_rate,
        citation_accuracy=citation_accuracy,
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        abstention_f1=_f1(abstention_precision, abstention_recall),
        mean_confidence_answerable=(
            sum(answerable_conf) / len(answerable_conf) if answerable_conf else 0.0
        ),
        mean_confidence_negative=(
            sum(negative_conf) / len(negative_conf) if negative_conf else 0.0
        ),
    )


def sweep_threshold(
    items: Sequence[EvalItem],
    service: AnswerLike,
    thresholds: Iterable[float],
    *,
    top_k: int | None = None,
) -> list[tuple[float, float, float, float]]:
    """Sweep the abstain threshold, running retrieval once per item.

    Retrieval is the expensive part, so each question is asked ONCE and the
    abstention decision is recomputed per candidate threshold.
    """
    observed: list[tuple[bool, float]] = []
    for item in items:
        answer = service.ask(item.question, top_k=top_k)
        observed.append((item.answerable, answer.confidence))

    n_negative = sum(1 for answerable, _ in observed if not answerable)
    out: list[tuple[float, float, float, float]] = []

    for threshold in thresholds:
        predicted = sum(1 for _, conf in observed if conf < threshold)
        true_positive = sum(
            1 for answerable, conf in observed if not answerable and conf < threshold
        )
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / n_negative if n_negative else 0.0
        out.append((threshold, precision, recall, _f1(precision, recall)))

    return out


def best_threshold(sweep: Sequence[tuple[float, float, float, float]]) -> float:
    """Highest-F1 threshold, breaking ties toward the lower value."""
    if not sweep:
        raise ValueError("cannot pick best threshold from empty sweep")
    return max(sweep, key=lambda entry: (entry[3], -entry[0]))[0]


def format_report(report: EvalReport) -> str:
    lines = [
        "Evaluation Report",
        "-----------------",
        f"Answerable items              : {report.n_answerable}",
        f"Negative items                : {report.n_negative}",
        f"Retrieval hit-rate            : {report.retrieval_hit_rate:.3f}",
        f"Citation accuracy (answered)  : {report.citation_accuracy:.3f}",
        f"Abstention precision          : {report.abstention_precision:.3f}",
        f"Abstention recall             : {report.abstention_recall:.3f}",
        f"Abstention F1                 : {report.abstention_f1:.3f}",
        f"Mean confidence (answerable)  : {report.mean_confidence_answerable:.3f}",
        f"Mean confidence (negative)    : {report.mean_confidence_negative:.3f}",
        "",
        f"{'id':30} {'hit':5} {'citation':9} {'abstained':10} {'conf':>6}",
        f"{'-' * 30} {'-' * 5} {'-' * 9} {'-' * 10} {'-' * 6}",
    ]
    for r in report.results:
        if r.item.answerable:
            hit = "hit" if r.retrieval_hit else "miss"
            citation = "ok" if r.citation_correct else "bad"
        else:
            hit = "n/a"
            citation = "n/a"
        abstained = "yes" if r.abstained else "no"
        lines.append(f"{r.item.id:30} {hit:5} {citation:9} {abstained:10} {r.confidence:6.3f}")
    return "\n".join(lines)
