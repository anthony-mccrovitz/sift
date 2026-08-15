"""Prove our id-based context metrics agree with RAGAS's own implementation.

The free retrieval gate computes context precision and recall itself rather than
importing RAGAS, because pulling the framework and the langchain stack into that
tier would add about ninety seconds of install to a gate whose whole value is
being cheap enough to run on every pull request including forks.

That is a reasonable trade and it carries an obvious risk: a reimplemented metric
can drift from the definition it claims to implement, and nothing would say so.
This file is the answer to that. It runs the real
`ragas.metrics.IDBasedContextPrecision` and `IDBasedContextRecall` over the same
inputs and asserts the numbers match.

Skipped automatically where RAGAS is not installed, which is the case in the
unit-test tier -- the point is that this runs *somewhere*, not everywhere. It
runs locally and in the RAGAS job, which is exactly where a disagreement would
matter.

    python -m pytest tests/test_ragas_agreement.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.retrieval_eval import id_based_context_scores  # noqa: E402

ragas_metrics = pytest.importorskip(
    "ragas.metrics", reason="RAGAS is not installed in this tier; see the module docstring"
)


# Real shapes from the eval set, plus the edge cases that break naive versions.
CASES = [
    # every retrieved chunk is from the gold document
    (["doc-a"] * 6, ["doc-a"]),
    # the usual case: several chunks of the gold document plus distractors
    (["doc-a", "doc-a", "doc-b", "doc-a", "doc-c", "doc-a"], ["doc-a"]),
    # a complete miss -- q017 looks like this
    (["doc-b", "doc-c", "doc-d", "doc-e", "doc-f", "doc-g"], ["doc-a"]),
    # more than one gold document, only one of them found
    (["doc-a", "doc-c", "doc-c", "doc-d", "doc-e", "doc-f"], ["doc-a", "doc-b"]),
    # more than one gold document, both found
    (["doc-a", "doc-b", "doc-c", "doc-a", "doc-e", "doc-f"], ["doc-a", "doc-b"]),
    # gold document arrives last
    (["doc-b", "doc-c", "doc-d", "doc-e", "doc-f", "doc-a"], ["doc-a"]),
]


def _ragas_scores(retrieved: list[str], gold: list[str]) -> tuple[float, float]:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._context_precision import IDBasedContextPrecision
    from ragas.metrics._context_recall import IDBasedContextRecall

    sample = SingleTurnSample(
        user_input="unused by an id-based metric",
        retrieved_context_ids=list(retrieved),
        reference_context_ids=list(gold),
    )
    precision = asyncio.run(IDBasedContextPrecision()._single_turn_ascore(sample, None))
    recall = asyncio.run(IDBasedContextRecall()._single_turn_ascore(sample, None))
    return precision, recall


@pytest.mark.parametrize("retrieved,gold", CASES)
def test_our_scores_match_ragas(retrieved, gold):
    ours_p, ours_r = id_based_context_scores(retrieved, gold)
    theirs_p, theirs_r = _ragas_scores(retrieved, gold)

    assert ours_p == pytest.approx(theirs_p, abs=1e-9), (
        f"context_precision disagrees for retrieved={retrieved} gold={gold}"
    )
    assert ours_r == pytest.approx(theirs_r, abs=1e-9), (
        f"context_recall disagrees for retrieved={retrieved} gold={gold}"
    )


def test_no_gold_documents_scores_zero_not_one():
    """Unanswerable questions must not score a free 1.0.

    Vacuous-truth definitions of precision would hand a perfect score to a
    question whose correct behaviour is to retrieve nothing, and the mean across
    the eval set would drift upward for the best possible reason and the worst
    possible cause. These questions are excluded from the mean entirely;
    abstention_rate is what measures them.
    """
    assert id_based_context_scores(["doc-a", "doc-b"], []) == (0.0, 0.0)


def test_retrieving_nothing_scores_zero_without_dividing_by_zero():
    assert id_based_context_scores([], ["doc-a"]) == (0.0, 0.0)


def test_passage_precision_is_deliberately_not_ragas_context_precision():
    """The two must differ when a document appears more than once.

    This is the disagreement that this file caught on its first run, kept as a
    test so the distinction cannot quietly collapse back into one number.
    """
    from eval.retrieval_eval import passage_precision

    retrieved = ["doc-a", "doc-b", "doc-c", "doc-a", "doc-e", "doc-f"]
    gold = ["doc-a", "doc-b"]

    ragas_style, _ = id_based_context_scores(retrieved, gold)
    ours = passage_precision(retrieved, gold)

    assert ragas_style == pytest.approx(2 / 5)  # distinct docs: a,b of a,b,c,e,f
    assert ours == pytest.approx(3 / 6)  # passages: a,b,a of six
    assert ragas_style != ours
