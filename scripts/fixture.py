"""Export / load the frozen evaluation corpus.

Why a fixture exists at all. CI cannot download 500 government PDFs and OCR them
on every pull request -- it would take an hour, cost real bandwidth on public
agency servers, and fail whenever gao.gov has a bad day. Worse, it would make
the eval non-deterministic: a metric that moved because the Federal Register
published new rules overnight tells you nothing about the diff under review.

So we freeze the parsed corpus. CI loads the exact same chunks every time, and
any change in the metrics is caused by the code in the pull request. That is the
whole point of a regression gate.

Embeddings are NOT stored -- they are recomputed at load time. The embedding
model is pinned in config.py, so recomputation is deterministic, and it keeps
the fixture ~20x smaller than storing 384 floats per chunk.

The fixture is a *sample* of the corpus, not all of it, because CI re-embeds
every chunk on load and the full corpus is too slow for a per-pull-request gate.
The sample always contains every gold document -- dropping one would make its
question unanswerable and quietly raise recall -- plus as many distractors as the
chunk budget allows. What the budget excludes is printed, never silent.

    python scripts/fixture.py export                  # budgeted (what CI uses)
    python scripts/fixture.py export --all            # entire corpus
    python scripts/fixture.py export --max-chunks N   # explicit budget
    python scripts/fixture.py load                    # in CI, or to rebuild locally
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sift.config import settings  # noqa: E402
from sift.db import connect, insert_chunks, upsert_document  # noqa: E402
from sift.embed import embed_passages  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "corpus.jsonl.gz"


def gold_doc_ids() -> set[str]:
    """Every document the eval set depends on. These can never be sampled out."""
    from eval.retrieval_eval import load_eval_set

    gold: set[str] = set()
    for question in load_eval_set():
        gold.update(question.get("gold_doc_ids") or [])
    return gold


def select_documents(conn, max_chunks: int | None) -> tuple[list[str], dict[str, int]]:
    """Choose which documents go in the fixture, under a chunk budget.

    The full corpus is ~50k chunks. CI re-embeds every chunk on load, which on a
    2-core runner is roughly eight minutes -- enough to push the job toward its
    timeout and make every pull request wait on it. So the fixture is a sample.

    Sampling a retrieval benchmark is dangerous in a specific way: drop the wrong
    document and recall goes *up*, because the question got easier. Two rules
    prevent that.

      1. Every gold document is included, always, budget or not. Without its gold
         document a question is unanswerable and recall silently measures nothing.
      2. Distractors are added in sorted doc_id order -- deterministic, so the
         fixture does not churn between exports and CI compares like with like.

    The distractors are the point. Recall@6 against 20 documents is arithmetic;
    against several hundred it is a measurement. The budget buys CI time, and
    what it costs is stated out loud rather than hidden -- see the printed
    summary and the README, which reports the full-corpus number separately.
    """
    rows = conn.execute(
        "SELECT doc_id, coalesce(chunk_count, 0) AS n FROM documents"
        " WHERE parse_status = 'parsed' ORDER BY doc_id"
    ).fetchall()
    sizes = {r["doc_id"]: r["n"] for r in rows}

    gold = gold_doc_ids()
    missing = sorted(g for g in gold if g not in sizes)
    if missing:
        print(f"  WARNING: {len(missing)} gold documents are not parsed: {missing}")

    selected = [d for d in sizes if d in gold]
    used = sum(sizes[d] for d in selected)

    distractors = [d for d in sizes if d not in gold]
    dropped = 0
    for doc_id in distractors:
        if max_chunks is not None and used + sizes[doc_id] > max_chunks:
            dropped += 1
            continue
        selected.append(doc_id)
        used += sizes[doc_id]

    stats = {
        "gold": sum(1 for d in selected if d in gold),
        "distractors": sum(1 for d in selected if d not in gold),
        "dropped": dropped,
        "chunks": used,
    }
    return sorted(selected), stats


def export_fixture(path: Path = FIXTURE, max_chunks: int | None = None) -> None:
    """Dump documents and chunk text (no vectors) to a gzipped JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with connect() as conn:
        selected, stats = select_documents(conn, max_chunks)

        with gzip.open(path, "wt", encoding="utf-8") as fh:
            docs = conn.execute(
                """
                SELECT doc_id, source, title, url, published_year, parse_status,
                       parse_strategy, is_scanned, page_count, element_count,
                       chunk_count, parse_seconds, parse_error
                FROM documents WHERE doc_id = ANY(%s) ORDER BY doc_id
                """,
                (selected,),
            ).fetchall()
            for doc in docs:
                fh.write(json.dumps({"type": "document", **dict(doc)}) + "\n")

            rows = conn.execute(
                """
                SELECT doc_id, chunk_index, text, page_start, page_end,
                       element_types, contains_table, char_count, doc_title
                FROM chunks WHERE doc_id = ANY(%s) ORDER BY doc_id, chunk_index
                """,
                (selected,),
            ).fetchall()
            for row in rows:
                fh.write(json.dumps({"type": "chunk", **dict(row)}) + "\n")

    size_mb = path.stat().st_size / 1_048_576
    print(
        f"Exported {len(docs)} documents ({stats['distractors']} distractors) "
        f"and {len(rows)} chunks -> {path} ({size_mb:.1f} MB)"
    )
    if stats["dropped"]:
        # Never let a cap pass silently. A fixture that quietly shrank is a
        # benchmark that quietly got easier.
        print(
            f"  {stats['dropped']} parsed documents excluded by the "
            f"{max_chunks}-chunk budget. Gold documents were never excluded."
        )


def load_fixture(path: Path = FIXTURE, batch_size: int = 256) -> None:
    """Recreate the corpus in Postgres, re-embedding chunk text as it goes."""
    if not path.exists():
        raise SystemExit(f"No fixture at {path}. Run: python scripts/fixture.py export")

    documents, chunks = [], []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            kind = record.pop("type")
            (documents if kind == "document" else chunks).append(record)

    print(f"Loading {len(documents)} documents and {len(chunks)} chunks")
    print(f"Re-embedding with {settings.embedding_model}...")

    with connect() as conn:
        conn.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
        for doc in documents:
            upsert_document(conn, doc)

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embed_passages([c["text"] for c in batch])
            for row, vector in zip(batch, vectors):
                row["embedding"] = vector
            insert_chunks(conn, batch)
            print(f"  {min(start + batch_size, len(chunks))}/{len(chunks)}")

        counts = conn.execute(
            "SELECT (SELECT count(*) FROM documents) AS d, (SELECT count(*) FROM chunks) AS c"
        ).fetchone()
    print(f"Loaded: {counts['d']} documents, {counts['c']} chunks")


# Chunk budget for the committed fixture. Chosen so CI's re-embedding step stays
# in the low minutes on a 2-core runner while leaving several hundred distractor
# documents in the index. Raise it if CI gets faster; lower it if the gate starts
# dominating pull-request time. Re-export and re-measure when you change it --
# the retrieval numbers move with the corpus, which is the whole point of it.
DEFAULT_MAX_CHUNKS = 14_000


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"export", "load"}:
        print(__doc__)
        return 1
    if sys.argv[1] == "export":
        max_chunks: int | None = DEFAULT_MAX_CHUNKS
        if "--all" in sys.argv:
            max_chunks = None
        elif "--max-chunks" in sys.argv:
            max_chunks = int(sys.argv[sys.argv.index("--max-chunks") + 1])
        export_fixture(max_chunks=max_chunks)
    else:
        load_fixture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
