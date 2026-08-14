"""Tier 2 evaluation: RAGAS metrics with an LLM judge.

Costs money and needs an API key, so it does not run on every push -- see
.github/workflows/eval.yml. What it buys that Tier 1 cannot: judgements about
the *generated answer* rather than about retrieval.

    faithfulness       is every claim in the answer supported by the context
    answer_relevancy   does the answer address the question asked
    context_precision  is the retrieved context on-topic (less noise is better)
    context_recall     does the retrieved context cover the ground truth

Note on the judge: the metrics use whatever LLM you configure, including the
same model that generated the answer. That is a real methodological weakness --
a model grading its own output is measurably more generous. Faithfulness is the
least affected (it is close to entailment checking) which is why it carries the
strictest threshold. Treat these as regression detectors, not absolute scores.

    python -m eval.ragas_eval --output benchmarks/latest_ragas.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sift.answer import answer_question  # noqa: E402
from sift.config import settings  # noqa: E402
from sift.llm import llm_available  # noqa: E402

from eval.retrieval_eval import (  # noqa: E402
    REPO_ROOT,
    load_eval_set,
    load_thresholds,
)


def build_judge():
    """Wrap the configured provider in RAGAS's LLM interface.

    RAGAS 0.4 takes any LangChain chat model through LangchainLLMWrapper, which
    is how we keep the provider-pluggable promise on the eval side too.
    """
    from ragas.llms import LangchainLLMWrapper

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return LangchainLLMWrapper(
            ChatAnthropic(
                model=settings.anthropic_model,
                temperature=0,
                max_tokens=1024,
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
        )

    from langchain_openai import ChatOpenAI

    return LangchainLLMWrapper(
        ChatOpenAI(model=settings.openai_model, temperature=0, api_key=os.environ["OPENAI_API_KEY"])
    )


def build_judge_embeddings():
    """answer_relevancy needs embeddings. Reuse the local model -- no reason to
    pay for an embedding API just to score an eval run."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model))


def collect_samples(questions: list[dict[str, Any]], skip_unanswerable: bool = True):
    """Run the real pipeline over the eval set and shape it for RAGAS."""
    from ragas import EvaluationDataset, SingleTurnSample

    samples, records = [], []
    for q in questions:
        # Unanswerable questions are excluded here on purpose. Faithfulness and
        # context_recall are undefined when the correct behaviour is to retrieve
        # nothing and say so -- Tier 1's abstention_rate is the right measure
        # for those, and mixing them in would drag the scores around for reasons
        # that have nothing to do with quality.
        if skip_unanswerable and not q.get("gold_doc_ids"):
            continue

        result = answer_question(q["question"])
        contexts = [h.text for h in result.hits] or ["(no context retrieved)"]

        samples.append(
            SingleTurnSample(
                user_input=q["question"],
                response=result.answer or "(no answer)",
                retrieved_contexts=contexts,
                reference=q["ground_truth"],
            )
        )
        records.append(
            {
                "id": q["id"],
                "category": q.get("category", "factual"),
                "question": q["question"],
                "answer": result.answer,
                "citations": [c.to_dict() for c in result.citations],
                "latency_ms": result.latency_ms,
            }
        )

    return EvaluationDataset(samples=samples), records


def main() -> int:
    parser = argparse.ArgumentParser(description="Sift RAGAS evaluation (needs an LLM key)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmarks" / "latest_ragas.json")
    parser.add_argument("--no-gate", action="store_true", help="report but always exit 0")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    args = parser.parse_args()

    if not llm_available():
        print(
            "No LLM configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.\n"
            "Tier 1 (python -m eval.retrieval_eval) runs without one."
        )
        return 1

    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    questions = load_eval_set()
    if args.limit:
        questions = questions[: args.limit]
    thresholds = load_thresholds()

    print(f"Running the pipeline over {len(questions)} questions...")
    dataset, records = collect_samples(questions)
    print(f"Scoring {len(dataset)} answerable questions with RAGAS "
          f"(judge: {settings.llm_provider})...\n")

    judge = build_judge()
    embeddings = build_judge_embeddings()
    metrics = [
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=embeddings),
        ContextPrecision(llm=judge),
        ContextRecall(llm=judge),
    ]

    result = evaluate(dataset=dataset, metrics=metrics, show_progress=True)

    # RAGAS emits NaN for samples it could not score; treat those as missing
    # rather than as zero, which would silently tank an otherwise fine run.
    scores: dict[str, float] = {}
    frame = result.to_pandas()
    for column in frame.columns:
        if column in {"user_input", "response", "retrieved_contexts", "reference"}:
            continue
        series = frame[column].dropna()
        if len(series):
            scores[column] = round(float(series.mean()), 4)

    print("\n" + "-" * 56)
    for name, value in scores.items():
        floor = thresholds.get("ragas", {}).get(name)
        mark = "" if floor is None else ("  PASS" if value >= floor else "  FAIL")
        target = "" if floor is None else f" (floor {floor})"
        print(f"  {name:22} {value:.4f}{target}{mark}")
    print("-" * 56)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scores": scores,
                "thresholds": thresholds.get("ragas", {}),
                "provider": settings.llm_provider,
                "judge_model": settings.anthropic_model
                if settings.llm_provider == "anthropic"
                else settings.openai_model,
                "questions_scored": len(dataset),
                "records": records,
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.output}")

    failures = [
        f"{name} {scores[name]:.4f} < required {floor}"
        for name, floor in thresholds.get("ragas", {}).items()
        if name in scores and scores[name] < floor
    ]
    if failures:
        print("\nQUALITY GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 0 if args.no_gate else 1

    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
