"""Tier 1 evaluation: deterministic, free, no LLM judge.

This is the gate that runs on every pull request. It measures the things that
can be checked by comparing to a known answer key rather than by asking a model
for an opinion:

    recall@k              did the gold document get retrieved at all
    MRR                   how high up
    citation validity     did every citation point at a real retrieved passage
    abstention            did the system decline when the answer is absent
    false abstention      did it decline when the answer was right there
    latency               p50 / p95 retrieval time

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

        if with_llm:
            answer = answer_question(q["question"], top_k=k)
            result.abstained = answer.abstained
            result.answered = not answer.abstained
            result.citation_count = len(answer.citations)
            result.invalid_citation_count = len(answer.invalid_citations)

        results.append(result)

    return results


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
    check_min("citation_validity")
    check_min("abstention_rate")

    # false_abstention_rate is a ceiling, not a floor.
    ceiling = retrieval.get("false_abstention_rate")
    if ceiling is not None and "false_abstention_rate" in metrics:
        if metrics["false_abstention_rate"] > ceiling:
            failures.append(
                f"false_abstention_rate {metrics['false_abstention_rate']:.4f} > allowed {ceiling}"
            )

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
    args = parser.parse_args()

    questions = load_eval_set()
    thresholds = load_thresholds()
    k = thresholds.get("retrieval", {}).get("k", settings.final_top_k)

    with_llm = llm_available() and not args.no_llm
    print(f"Evaluating {len(questions)} questions at k={k}")
    print(f"LLM judge/answering: {'enabled' if with_llm else 'DISABLED (retrieval metrics only)'}\n")

    results = evaluate_retrieval(questions, k=k, with_llm=with_llm)
    metrics = summarise(results, with_llm=with_llm)

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

    failures = check_thresholds(metrics, thresholds)
    if failures:
        print("\nQUALITY GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 0 if args.no_gate else 1

    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
