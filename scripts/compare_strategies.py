"""Compare Unstructured parse strategies on the same PDF, and score the result.

Written because the scanned NASA documents came out of hi_res with correct words
in scrambled order, and "which strategy is least bad on a 1967 microfiche scan"
is a question worth answering with numbers rather than an opinion.

    python scripts/compare_strategies.py data/raw/nasa/nasa-19670021251.pdf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Run from anywhere without needing PYTHONPATH set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sift.ingest.chunk import chunk_elements  # noqa: E402
from sift.ingest.partition import (  # noqa: E402
    STRATEGY_FAST,
    STRATEGY_HI_RES,
    STRATEGY_OCR,
    _garbled_score,
    partition_document,
)

# A crude readability proxy: real English text has function words in the right
# places. Scrambled OCR keeps the words but destroys the bigrams, so counting
# common English bigrams separates "readable" from "word salad" better than
# any per-word check.
COMMON_BIGRAMS = {
    "of the", "in the", "to the", "is a", "for the", "on the", "and the",
    "it is", "this is", "that the", "with the", "from the", "by the",
    "as a", "at the", "to be", "may be", "can be", "such as", "which is",
    "the same", "the first", "has been", "have been", "will be", "the results",
}


def bigram_score(text: str) -> float:
    """Common English bigrams per 1000 words. Higher is more readable."""
    words = text.lower().split()
    if len(words) < 50:
        return 0.0
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    hits = sum(1 for b in bigrams if b in COMMON_BIGRAMS)
    return round(hits / len(words) * 1000, 2)


def main(path_str: str) -> int:
    path = Path(path_str)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    print(f"Comparing parse strategies on {path.name}\n")
    print(f"{'strategy':10} {'seconds':>8} {'elements':>9} {'chunks':>7} {'chars':>8} {'garbled':>8} {'bigrams/1k':>11}")
    print("-" * 70)

    results = {}
    for strategy in (STRATEGY_FAST, STRATEGY_HI_RES, STRATEGY_OCR):
        started = time.monotonic()
        parsed = partition_document(path, timeout=1800, force_strategy=strategy)
        elapsed = time.monotonic() - started

        if parsed.status == "failed":
            print(f"{strategy:10} {elapsed:>8.1f}  FAILED: {parsed.error}")
            continue

        chunks = chunk_elements(parsed.elements, "compare")
        text = "\n".join(c["text"] for c in chunks)
        results[strategy] = {
            "seconds": round(elapsed, 1),
            "elements": parsed.element_count,
            "chunks": len(chunks),
            "chars": len(text),
            "garbled": round(_garbled_score(text), 2),
            "bigrams": bigram_score(text),
            "sample": text[:400],
        }
        r = results[strategy]
        print(
            f"{strategy:10} {r['seconds']:>8.1f} {r['elements']:>9} {r['chunks']:>7} "
            f"{r['chars']:>8} {r['garbled']:>8.2f} {r['bigrams']:>11.2f}"
        )

    print("\n" + "=" * 70)
    for strategy, r in results.items():
        print(f"\n--- {strategy} sample ---\n{r['sample']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/nasa/nasa-19670021251.pdf"))
