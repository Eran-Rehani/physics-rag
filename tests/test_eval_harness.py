from __future__ import annotations

from pathlib import Path

import pytest

from physics_rag.answer import Answer
from physics_rag.evaluation import (
    EvalItem,
    best_threshold,
    format_report,
    load_eval_set,
    match_citation,
    normalise_filename,
    run_eval,
    score_item,
    sweep_threshold,
)
from physics_rag.generation import GenerationError
from physics_rag.store import SearchResult


class FakeAnswerService:
    def __init__(self, answers: dict[str, Answer]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        self.calls.append(question)
        return self.answers[question]


def _result(filename: str, section_path: str = "", page: int | None = None) -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk-{filename}",
        text=f"text from {filename}",
        score=0.8,
        filename=filename,
        section_path=section_path,
        page=page,
        math_fidelity="exact",
        source_path=f"/data/{filename}",
    )


def _answer(
    question: str,
    text: str,
    results: list[SearchResult],
    confidence: float = 0.8,
    abstained: bool = False,
) -> Answer:
    return Answer(
        question=question,
        text=text,
        abstained=abstained,
        confidence=confidence,
        citations=[],
        results=results,
    )


def _write(tmp_path: Path, content: str, name: str = "eval.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_eval_set_parses_section_and_page_forms(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
questions:
  - id: q1
    question: "What?"
    answerable: true
    expected_files:
      - Lec_03.tex
    expected_citations:
      - file: Lec_03.tex
        section: "אנטרופיה"
  - id: q2
    question: "Another?"
    answerable: false
    expected_files: []
    expected_citations:
      - file: PartitionFunction2025.pdf
        page: 1
""",
    )

    items = load_eval_set(path)

    assert [item.id for item in items] == ["q1", "q2"]
    assert items[0].expected_citations == (("Lec_03.tex", "אנטרופיה", None),)
    assert items[1].answerable is False
    assert items[1].expected_citations == (("PartitionFunction2025.pdf", None, 1),)


@pytest.mark.parametrize(
    ("name", "body"),
    [
        (
            "duplicate",
            "questions:\n"
            '  - {id: a, question: "one", expected_files: [x.tex]}\n'
            '  - {id: a, question: "two", expected_files: [x.tex]}\n',
        ),
        ("missing_question", "questions:\n  - {id: a, expected_files: [x.tex]}\n"),
        ("answerable_no_files", 'questions:\n  - {id: a, question: "q", expected_files: []}\n'),
        (
            "negative_with_files",
            'questions:\n  - {id: a, question: "q", answerable: false, expected_files: [x.tex]}\n',
        ),
    ],
)
def test_load_eval_set_rejects_malformed(tmp_path: Path, name: str, body: str) -> None:
    with pytest.raises(ValueError):
        load_eval_set(_write(tmp_path, body, f"{name}.yaml"))


def test_match_citation_variants() -> None:
    assert match_citation(
        "[Lec_03.tex, שבוע 2 > אנטרופיה של מוצק]", ("Lec_03.tex", "אנטרופיה", None)
    )
    assert match_citation("[Partition.pdf, p. 1]", ("Partition.pdf", None, 1))
    assert match_citation("[notes.tex]", ("notes.tex", None, None))
    # Section parts may themselves contain commas: split on the FIRST comma only.
    assert match_citation("[notes.tex, חומר, חלק א]", ("notes.tex", "חלק א", None))
    assert not match_citation("[other.tex, section]", ("notes.tex", "section", None))
    assert not match_citation("[notes.pdf, p. 2]", ("notes.pdf", None, 1))


def test_normalise_filename_equates_bare_name_and_path() -> None:
    assert normalise_filename("/some/path/Lec_03.tex") == normalise_filename("Lec_03.tex")
    assert normalise_filename("LEC_03.TEX") == normalise_filename("lec_03.tex")


def test_score_item_uses_expected_citations() -> None:
    item = EvalItem(
        id="q1",
        question="What?",
        expected_files=("Lec_03.tex",),
        expected_citations=(("Lec_03.tex", "אנטרופיה", None),),
    )
    answer = _answer(
        item.question,
        "The entropy is ... [Lec_03.tex, שבוע 2 > אנטרופיה של מוצק]",
        [_result("Lec_03.tex", section_path="שבוע 2 > אנטרופיה של מוצק")],
    )

    scored = score_item(item, answer)

    assert scored.retrieval_hit is True
    assert scored.citation_correct is True
    assert scored.correct_abstention is True


def test_score_item_falls_back_to_expected_files() -> None:
    item = EvalItem(id="q2", question="How?", expected_files=("notes.tex",))
    answer = _answer(item.question, "See [notes.tex, Some Section]", [_result("notes.tex")])

    scored = score_item(item, answer)

    assert scored.retrieval_hit is True
    assert scored.citation_correct is True


def test_score_item_negative_scores_abstention_only() -> None:
    item = EvalItem(id="neg", question="Q", answerable=False)
    answer = _answer("Q", "not found in corpus", [], confidence=0.1, abstained=True)

    scored = score_item(item, answer)

    assert scored.retrieval_hit is False
    assert scored.correct_abstention is True


def test_run_eval_computes_obvious_metrics() -> None:
    items = [
        EvalItem(id="a1", question="q1", expected_files=("a.tex",)),
        EvalItem(id="a2", question="q2", expected_files=("b.tex",)),
        EvalItem(id="n1", question="q3", answerable=False),
        EvalItem(id="n2", question="q4", answerable=False),
    ]
    answers = {
        "q1": _answer("q1", "Answer [a.tex, sec]", [_result("a.tex", "sec")], confidence=0.8),
        "q2": _answer(
            "q2", "Answer [b.tex, sec]", [_result("b.tex", "sec")], confidence=0.3, abstained=True
        ),
        "q3": _answer("q3", "not found in corpus", [], confidence=0.2, abstained=True),
        "q4": _answer("q4", "something", [_result("d.tex")], confidence=0.6, abstained=False),
    }
    service = FakeAnswerService(answers)

    report = run_eval(items, service)

    assert report.n_answerable == 2
    assert report.n_negative == 2
    assert report.retrieval_hit_rate == 1.0
    assert report.citation_accuracy == 1.0
    # Abstained: a2 (wrongly) and n1 (rightly) -> precision 1/2.
    assert report.abstention_precision == 0.5
    # Negatives: n1 abstained, n2 did not -> recall 1/2.
    assert report.abstention_recall == 0.5
    assert report.abstention_f1 == 0.5
    assert report.mean_confidence_answerable == pytest.approx(0.55)
    assert report.mean_confidence_negative == pytest.approx(0.4)
    assert "Retrieval hit-rate" in format_report(report)


def test_sweep_threshold_asks_each_question_exactly_once() -> None:
    items = [
        EvalItem(id="a", question="q1", expected_files=("x.tex",)),
        EvalItem(id="b", question="q2", answerable=False),
        EvalItem(id="c", question="q3", answerable=False),
    ]
    answers = {
        "q1": _answer("q1", "x", [], confidence=0.8),
        "q2": _answer("q2", "x", [], confidence=0.4),
        "q3": _answer("q3", "x", [], confidence=0.1),
    }
    service = FakeAnswerService(answers)
    thresholds = [0.0, 0.2, 0.5, 0.7, 0.9]

    sweep = sweep_threshold(items, service, thresholds)

    # Retrieval is the expensive part: five thresholds must not mean five passes.
    assert len(service.calls) == 3
    assert len(sweep) == len(thresholds)

    # At 0.5 both negatives abstain and the answerable one does not: perfect.
    at_half = next(entry for entry in sweep if entry[0] == 0.5)
    assert at_half[1] == 1.0
    assert at_half[2] == 1.0
    assert at_half[3] == 1.0


def test_best_threshold_breaks_ties_toward_lower_value() -> None:
    sweep = [
        (0.2, 0.5, 0.5, 0.5),
        (0.4, 0.8, 0.8, 0.8),
        (0.6, 0.8, 0.8, 0.8),
        (0.8, 0.6, 0.6, 0.6),
    ]

    assert best_threshold(sweep) == 0.4


def test_best_threshold_rejects_empty_sweep() -> None:
    with pytest.raises(ValueError):
        best_threshold([])


class RaisingAnswerService:
    """Answers normally except for the questions listed in ``failing``."""

    def __init__(self, answers: dict[str, Answer], failing: set[str]) -> None:
        self.answers = answers
        self.failing = failing

    def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        if question in self.failing:
            raise GenerationError("generation hit the 1024-token limit and was truncated")
        return self.answers[question]


def test_run_eval_survives_a_generation_failure_and_counts_it() -> None:
    items = [
        EvalItem(id="a1", question="q1", expected_files=("a.tex",)),
        EvalItem(id="a2", question="q2", expected_files=("b.tex",)),
    ]
    answers = {
        "q1": _answer("q1", "Answer [a.tex, sec]", [_result("a.tex", "sec")], confidence=0.8),
    }
    service = RaisingAnswerService(answers, failing={"q2"})

    report = run_eval(items, service)

    assert report.n_generation_failures == 1
    # The failure must not land in the citation denominator: one answered item,
    # cited correctly, so accuracy stays 1.0 rather than being diluted to 0.5.
    assert report.citation_accuracy == 1.0
    failed = next(r for r in report.results if r.item.id == "a2")
    assert failed.generation_error
    assert failed.citation_correct is False
    assert "ERROR" in format_report(report)
    assert "Generation failures           : 1" in format_report(report)


def test_sweep_threshold_skips_questions_the_generator_could_not_answer() -> None:
    items = [
        EvalItem(id="a", question="q1", expected_files=("x.tex",)),
        EvalItem(id="n", question="q2", answerable=False),
    ]
    answers = {"q1": _answer("q1", "text", [_result("x.tex")], confidence=0.9)}
    service = RaisingAnswerService(answers, failing={"q2"})

    sweep = sweep_threshold(items, service, [0.5])

    # The only negative failed, so there is nothing to abstain on and the sweep
    # still returns rather than aborting the whole run.
    assert len(sweep) == 1
    assert sweep[0][0] == 0.5
