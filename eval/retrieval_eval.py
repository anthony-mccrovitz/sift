"""Tier 1 evaluation: deterministic, free, no LLM judge.

This is the gate that runs on every pull request. It measures the things that
can be checked by comparing to a known answer key rather than by asking a model
for an opinion:

    recall@k              did the gold document get retrieved at all
    MRR                   how high up
    context precision     of the distinct documents retrieved, how many were
                          relevant (RAGAS IDBasedContextPrecision, no judge)
    context recall        of the gold documents, how many were retrieved
                          (RAGAS IDBasedContextRecall, no judge)
    passage precision     of the k passages given to the model, how many came
                          from a gold document
    citation validity     did every citation point at a real retrieved passage
    abstention            did the system decline when the answer is absent
    false abstention      did it decline when the answer was right there
    latency               p50 / p95 retrieval time
    determinism           does an identical second run return identical results

Retrieval metrics need no LLM at all. Citation and abstention metrics do call
the LLM (they are properties of a generated answer), so they are skipped
automatically when no key is present -- the retrieval half still gates.

    python -m eval.retrieval_eval --output benchmarks/latest_retrieval.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sift.answer import answer_question  # noqa: E402
from sift.config import settings  # noqa: E402
from sift.db import session_settings  # noqa: E402
from sift.llm import llm_available  # noqa: E402
from sift.retrieval.search import hybrid_search  # noqa: E402
from sift.retrieval.router import route  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = REPO_ROOT / "eval" / "eval_set.yaml"
THRESHOLDS = REPO_ROOT / "config" / "thresholds.yaml"


@dataclass
class QuestionResult:
    id: str
    question: str
    category: str
    gold_doc_ids: list[str]
    retrieved_doc_ids: list[str] = field(default_factory=list)
    hit: bool = False
    reciprocal_rank: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    passage_precision: float = 0.0
    retrieval_ms: int = 0
    routing_mode: str = ""
    # Only populated when an LLM is configured.
    answered: bool | None = None
    abstained: bool | None = None
    citation_count: int = 0
    invalid_citation_count: int = 0


def load_eval_set(path: Path = EVAL_SET) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    questions = data.get("questions", [])
    if not questions:
        raise SystemExit(f"No questions found in {path}")
    return questions


def load_thresholds(path: Path = THRESHOLDS) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def evaluate_retrieval(questions: list[dict], k: int, with_llm: bool) -> list[QuestionResult]:
    results: list[QuestionResult] = []

    # Warm the embedding model before timing anything. sentence-transformers
    # loads lazily, so without this the first question absorbs ~2.5s of model
    # load and lands in the p95 -- turning a latency gate into a coin flip on
    # whether CI's disk cache was warm. Measure the system, not the import.
    from sift.embed import embed_query

    embed_query("warmup")

    # The database needs no equivalent warm-up pass here, and it is worth saying
    # why, because the obvious guess is wrong. CI evaluates immediately after a
    # bulk load, and roughly one clean build in four used to fail on latency at
    # p50 330ms against a 250ms budget. That is not a cold cache -- every query
    # was slow, not just the first -- it was autovacuum waking up on a freshly
    # written table and competing with the queries being timed. A warm-up pass
    # here was tried and did not fix it. The fix is in the loader, which now
    # does that housekeeping itself: scripts/fixture.py, load_fixture.

    for q in questions:
        gold = q.get("gold_doc_ids") or []
        decision = route(q["question"])

        started = time.monotonic()
        hits = hybrid_search(
            q["question"],
            top_k=k,
            filters=decision.filters,
            use_vector=decision.use_vector,
            use_keyword=decision.use_keyword,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        retrieved = [h.doc_id for h in hits]
        result = QuestionResult(
            id=q["id"],
            question=q["question"],
            category=q.get("category", "factual"),
            gold_doc_ids=gold,
            retrieved_doc_ids=retrieved,
            retrieval_ms=elapsed_ms,
            routing_mode=decision.mode,
        )

        # Rank of the first gold document. Unanswerable questions have no gold
        # documents and are excluded from recall/MRR entirely -- scoring them
        # would mean rewarding retrieval for finding nothing.
        if gold:
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in gold:
                    result.hit = True
                    result.reciprocal_rank = 1.0 / rank
                    break
            result.context_precision, result.context_recall = id_based_context_scores(
                retrieved, gold
            )
            result.passage_precision = passage_precision(retrieved, gold)

        if with_llm:
            answer = answer_question(q["question"], top_k=k)
            result.abstained = answer.abstained
            result.answered = not answer.abstained
            result.citation_count = len(answer.citations)
            result.invalid_citation_count = len(answer.invalid_citations)

        results.append(result)

    return results


# Forces sequential scans onto parallel workers, which changes the order tied
# rows arrive in and therefore which of them survives a LIMIT. The point is to
# make the repeat pass run under a *different plan*, not merely a second time.
_PERTURBED_PLANNER = {
    "max_parallel_workers_per_gather": "4",
    "parallel_setup_cost": "0",
    "parallel_tuple_cost": "0",
    "min_parallel_table_scan_size": "0",
}


def id_based_context_scores(retrieved: list[str], gold: list[str]) -> tuple[float, float]:
    """Context precision and recall over document ids. No LLM, no key.

    These are two of the four RAGAS metrics, and they are the two that never
    needed a judge. RAGAS ships `IDBasedContextPrecision` and
    `IDBasedContextRecall` for exactly this case, and both are set arithmetic
    over ids rather than a model's opinion.

    Computed here rather than by importing RAGAS, because pulling the framework
    and the langchain stack into the *free* tier would add about ninety seconds
    of install to a gate whose entire value is being cheap enough to run on every
    pull request including forks. Agreement is verified rather than assumed --
    tests/test_ragas_agreement.py runs RAGAS's own implementation over the same
    inputs and asserts the numbers match.

    That test earned itself immediately. My first version divided by the number
    of retrieved *passages*; RAGAS deduplicates to distinct document ids first,
    so the two disagreed (0.5 against 0.4) the moment a document appeared twice
    in the results -- which, with six chunks drawn from a handful of documents,
    is most of the time. RAGAS's definition is the one that ships under RAGAS's
    name. The passage-level number is still worth having and still reported, as
    `passage_precision`, because it answers a different question.
    """
    if not gold or not retrieved:
        # Unanswerable questions have no gold documents. Scoring them would
        # reward retrieving nothing; abstention_rate is their measure.
        return 0.0, 0.0

    retrieved_set, gold_set = set(retrieved), set(gold)
    precision = len(retrieved_set & gold_set) / len(retrieved_set)
    recall = len(gold_set & retrieved_set) / len(gold_set)
    return precision, recall


def passage_precision(retrieved: list[str], gold: list[str]) -> float:
    """What fraction of the k passages handed to the model came from a gold doc.

    Deliberately not RAGAS's context_precision, which deduplicates to distinct
    documents. Both are worth knowing and they answer different questions:

        context_precision   of the distinct documents retrieved, how many were
                            relevant -- document-level noise
        passage_precision   of the six passages actually occupying the context
                            window, how many were on-target -- how much of the
                            model's attention budget was spent well

    The deduplicated version is also jumpy at this scale: six chunks from the
    gold document alone score 1.0, and adding a single distractor document
    halves it to 0.5, which is a large move for a small change.
    """
    if not gold or not retrieved:
        return 0.0
    gold_set = set(gold)
    return sum(doc_id in gold_set for doc_id in retrieved) / len(retrieved)


def check_determinism(questions: list[dict], k: int, baseline: list[QuestionResult]) -> list[str]:
    """Re-run retrieval under a different query plan and report what changed.

    This exists because the gate was, for a while, quietly lying. Both retrievers
    ended in `ORDER BY <score> LIMIT k` with no tiebreak, and ts_rank_cd ties
    constantly, so which of the tied rows survived the LIMIT was the executor's
    choice -- stable enough to look fine, unstable enough that the same corpus
    and the same code produced MRR 0.9115 on one run and 0.9125 on the next.

    A regression gate that moves on its own is worse than no gate. It spends the
    team's trust on false reds, and it hides real regressions inside its own
    noise floor.

    Note what this check is *not*: simply running the eval twice. That was the
    first version of it, and it passed happily with the bug reintroduced --
    consecutive runs pick the same plan, so they agree with each other while both
    remain arbitrary. Two runs agreeing is not evidence of determinism; it is
    evidence that nothing perturbed them. So the repeat pass deliberately runs
    under different planner settings, and the question it asks is the one that
    actually matters: does the answer depend on how Postgres chose to execute it?

    Retrieval only, so it costs one extra pass over the question set (a few
    seconds) and never touches the LLM.

    One limit worth stating: the perturbation only bites if the server can
    actually launch parallel workers. Where it cannot, the repeat pass degrades
    to a plain second run and the check gets weaker rather than wrong -- it can
    still fail, it just has fewer ways to.
    """
    with session_settings(**_PERTURBED_PLANNER):
        repeat = evaluate_retrieval(questions, k=k, with_llm=False)
    by_id = {r.id: r for r in repeat}

    failures = []
    for first in baseline:
        second = by_id.get(first.id)
        if second and first.retrieved_doc_ids != second.retrieved_doc_ids:
            failures.append(
                f"{first.id} retrieved different documents under a different query plan\n"
                f"      run 1: {first.retrieved_doc_ids}\n"
                f"      run 2: {second.retrieved_doc_ids}"
            )
    return failures


def summarise(results: list[QuestionResult], with_llm: bool) -> dict[str, Any]:
    answerable = [r for r in results if r.gold_doc_ids]
    unanswerable = [r for r in results if not r.gold_doc_ids]

    latencies = [r.retrieval_ms for r in results]
    metrics: dict[str, Any] = {
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "recall_at_k": round(sum(r.hit for r in answerable) / max(1, len(answerable)), 4),
        "mrr": round(sum(r.reciprocal_rank for r in answerable) / max(1, len(answerable)), 4),
        "id_based_context_precision": round(
            sum(r.context_precision for r in answerable) / max(1, len(answerable)), 4
        ),
        "id_based_context_recall": round(
            sum(r.context_recall for r in answerable) / max(1, len(answerable)), 4
        ),
        "passage_precision": round(
            sum(r.passage_precision for r in answerable) / max(1, len(answerable)), 4
        ),
        "retrieval_p50_ms": int(statistics.median(latencies)) if latencies else 0,
        "retrieval_p95_ms": int(
            statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies, default=0)
        ),
    }

    # Per-category recall makes regressions legible: "OCR questions dropped"
    # is actionable, "recall dropped" is not.
    by_category: dict[str, list[QuestionResult]] = {}
    for r in answerable:
        by_category.setdefault(r.category, []).append(r)
    metrics["recall_by_category"] = {
        cat: round(sum(r.hit for r in rs) / len(rs), 4) for cat, rs in sorted(by_category.items())
    }

    if with_llm:
        total_citations = sum(r.citation_count for r in results)
        invalid = sum(r.invalid_citation_count for r in results)
        metrics["citations_emitted"] = total_citations
        metrics["citations_invalid"] = invalid
        metrics["citation_validity"] = round(
            (total_citations - invalid) / total_citations, 4
        ) if total_citations else 1.0

        if unanswerable:
            metrics["abstention_rate"] = round(
                sum(bool(r.abstained) for r in unanswerable) / len(unanswerable), 4
            )
        if answerable:
            metrics["false_abstention_rate"] = round(
                sum(bool(r.abstained) for r in answerable) / len(answerable), 4
            )

    return metrics


def check_thresholds(metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Return a list of human-readable failures. Empty means the gate passes."""
    failures: list[str] = []
    retrieval = thresholds.get("retrieval", {})
    latency = thresholds.get("latency", {})

    def check_min(name: str, floor_key: str | None = None) -> None:
        floor = retrieval.get(floor_key or name)
        if floor is None or name not in metrics:
            return
        if metrics[name] < floor:
            failures.append(f"{name} {metrics[name]:.4f} < required {floor}")

    check_min("recall_at_k")
    check_min("mrr")
    check_min("id_based_context_precision")
    check_min("id_based_context_recall")
    check_min("passage_precision")
    check_min("citation_validity")
    check_min("abstention_rate")

    # false_abstention_rate is a ceiling, not a floor.
    ceiling = retrieval.get("false_abstention_rate")
    if ceiling is not None and "false_abstention_rate" in metrics:
        if metrics["false_abstention_rate"] > ceiling:
            failures.append(
                f"false_abstention_rate {metrics['false_abstention_rate']:.4f} > allowed {ceiling}"
            )

    # Latency is not gated when the generator is running on this machine, and
    # this is a real exemption rather than a convenient one, so it is stated
    # rather than silently applied -- main() prints it every time it engages.
    #
    # The local provider holds a language model on the same GPU the query
    # encoder uses. Measured on the identical corpus and code, retrieval p50
    # went from 216ms to 606ms purely from that contention. Gating on it would
    # fail the build for owning one GPU, which tells you nothing about the pull
    # request. CI never hits this: it has no key and no local model, so it
    # measures retrieval on its own.
    if settings.llm_provider == "local":
        return failures

    for key in ("retrieval_p50_ms", "retrieval_p95_ms"):
        budget = latency.get(key)
        if budget is not None and metrics.get(key, 0) > budget:
            failures.append(f"{key} {metrics[key]}ms > budget {budget}ms")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Sift retrieval evaluation (no LLM judge)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmarks" / "latest_retrieval.json")
    parser.add_argument("--no-llm", action="store_true", help="skip answer-level metrics even if a key exists")
    parser.add_argument("--no-gate", action="store_true", help="report metrics but always exit 0")
    parser.add_argument(
        "--skip-determinism",
        action="store_true",
        help="skip the repeat-run determinism check (it costs one extra retrieval pass)",
    )
    args = parser.parse_args()

    questions = load_eval_set()
    thresholds = load_thresholds()
    k = thresholds.get("retrieval", {}).get("k", settings.final_top_k)

    with_llm = llm_available() and not args.no_llm
    print(f"Evaluating {len(questions)} questions at k={k}")
    print(f"LLM judge/answering: {'enabled' if with_llm else 'DISABLED (retrieval metrics only)'}\n")

    results = evaluate_retrieval(questions, k=k, with_llm=with_llm)

    nondeterminism: list[str] = []
    if not args.skip_determinism:
        nondeterminism = check_determinism(questions, k=k, baseline=results)

    metrics = summarise(results, with_llm=with_llm)
    metrics["deterministic"] = None if args.skip_determinism else not nondeterminism

    print("-" * 62)
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub, sub_value in value.items():
                print(f"      {sub:16} {sub_value}")
        else:
            print(f"  {key:24} {value}")
    print("-" * 62)

    # Always show what missed -- a number without the failing cases is not
    # actionable when CI goes red.
    misses = [r for r in results if r.gold_doc_ids and not r.hit]
    if misses:
        print(f"\n{len(misses)} question(s) did not retrieve a gold document:")
        for r in misses:
            print(f"  [{r.category}] {r.id} {r.question[:56]}")
            print(f"      expected one of {r.gold_doc_ids}")
            print(f"      got {r.retrieved_doc_ids[:4]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "thresholds": thresholds,
                "with_llm": with_llm,
                "embedding_model": settings.embedding_model,
                "results": [r.__dict__ for r in results],
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.output}")

    if settings.llm_provider == "local":
        print(
            "\nNOTE: latency thresholds are not enforced in this run. The local\n"
            "      generator shares a GPU with the query encoder, which inflates\n"
            "      retrieval latency roughly 3x (measured: 216ms -> 606ms p50).\n"
            "      Quote latency from a run without a local generator.\n"
        )

    failures = check_thresholds(metrics, thresholds)
    if nondeterminism:
        # Listed first: every other number in this report is unreliable while
        # this is true, so fixing it comes before reading them.
        failures = [
            f"retrieval is not deterministic -- {len(nondeterminism)} question(s) "
            f"returned different documents under a different query plan",
            *nondeterminism,
        ] + failures
    if failures:
        print("\nQUALITY GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 0 if args.no_gate else 1

    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
