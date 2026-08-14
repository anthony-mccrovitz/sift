"""Append the current metrics to benchmarks/history.md and refresh the README.

Run after an evaluation. Keeping the README's results table generated rather
than hand-written is the only reliable way to stop it drifting into fiction --
a README claiming 0.97 recall while the code scores 0.81 is worse than one with
no numbers at all.

    python -m eval.retrieval_eval
    python scripts/record_benchmark.py --note "switched scans to ocr_only"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY = REPO_ROOT / "benchmarks" / "history.md"
README = REPO_ROOT / "README.md"
RETRIEVAL = REPO_ROOT / "benchmarks" / "latest_retrieval.json"
RAGAS = REPO_ROOT / "benchmarks" / "latest_ragas.json"

HEADER = """# Benchmark history

Appended by `scripts/record_benchmark.py` after each evaluation run. The point
is to catch slow drift -- a gate only fails on a single bad pull request, while
a table shows quality bleeding away over ten good-looking ones.

| date | commit | docs | chunks | recall@6 | MRR | p50 ms | faithfulness | note |
|---|---|---|---|---|---|---|---|---|
"""


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- not a git checkout, or git missing
        return "unknown"


def corpus_counts() -> tuple[str, str]:
    """Documents parsed / attempted, and chunk count, straight from the DB."""
    try:
        from sift.db import connect

        with connect() as conn:
            row = conn.execute(
                """
                SELECT (SELECT count(*) FROM documents) AS total,
                       (SELECT count(*) FROM documents WHERE parse_status='parsed') AS parsed,
                       (SELECT count(*) FROM chunks) AS chunks
                """
            ).fetchone()
        return f"{row['parsed']}/{row['total']}", str(row["chunks"])
    except Exception:  # noqa: BLE001 -- database not running
        return "n/a", "n/a"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="what changed since the last run")
    args = parser.parse_args()

    if not RETRIEVAL.exists():
        print(f"No metrics at {RETRIEVAL}. Run: python -m eval.retrieval_eval")
        return 1

    metrics = json.loads(RETRIEVAL.read_text())["metrics"]
    faithfulness = "-"
    if RAGAS.exists():
        faithfulness = str(json.loads(RAGAS.read_text()).get("scores", {}).get("faithfulness", "-"))

    docs, chunks = corpus_counts()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = (
        f"| {date} | `{git_sha()}` | {docs} | {chunks} | "
        f"{metrics.get('recall_at_k', '-')} | {metrics.get('mrr', '-')} | "
        f"{metrics.get('retrieval_p50_ms', '-')} | {faithfulness} | {args.note} |\n"
    )

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY.exists():
        HISTORY.write_text(HEADER)
    with HISTORY.open("a") as fh:
        fh.write(row)
    print(f"Appended to {HISTORY}:\n  {row.strip()}")

    # --- refresh the README results table -------------------------------
    table = (
        "<!-- RESULTS:START -->\n"
        "| Metric | Score |\n|---|---|\n"
        f"| Recall@{metrics.get('k', 6)} | **{metrics.get('recall_at_k', '-')}** |\n"
        f"| MRR | **{metrics.get('mrr', '-')}** |\n"
        f"| Documents parsed | **{docs}** |\n"
        f"| Chunks indexed | {chunks} |\n"
        f"| Median retrieval latency | **{metrics.get('retrieval_p50_ms', '-')} ms** |\n"
        f"| p95 retrieval latency | {metrics.get('retrieval_p95_ms', '-')} ms |\n"
    )
    if "citation_validity" in metrics:
        table += f"| Citation validity | **{metrics['citation_validity']}** |\n"
    if "abstention_rate" in metrics:
        table += f"| Correct abstention (unanswerable) | {metrics['abstention_rate']} |\n"
    if faithfulness != "-":
        table += f"| RAGAS faithfulness | **{faithfulness}** |\n"
    table += "<!-- RESULTS:END -->"

    readme = README.read_text()
    updated = re.sub(
        r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", table, readme, flags=re.S
    )
    if updated != readme:
        README.write_text(updated)
        print("Updated the README results table.")
    else:
        print("README markers not found -- results table left unchanged.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
