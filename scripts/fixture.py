"""Export / load the frozen evaluation corpus.

Why a fixture exists at all. CI cannot download 500 government PDFs and OCR them
on every pull request -- it would take an hour, cost real bandwidth on public
agency servers, and fail whenever gao.gov has a bad day. Worse, it would make
the eval non-deterministic: a metric that moved because the Federal Register
published new rules overnight tells you nothing about the diff under review.

So we freeze the parsed corpus. CI loads the exact same chunks every time, and
any change in the metrics is caused by the code in the pull request. That is the
whole point of a regression gate.

Embeddings ARE stored, as float16. They did not used to be: the model is pinned
in config.py, so recomputing them at load time was deterministic and kept the
file far smaller. Then the gate was measured rather than estimated. A GitHub
runner embeds ~9.5 chunks/sec, so re-embedding 14k chunks took 24m34s of a
26m32s job -- 93% of the gate spent recomputing a pure function of committed
text. Storing the vectors trades repository size for that, and the alternative
(shrinking the fixture until it fit) would have cut it to ~33 documents and
destroyed the distractor set the benchmark depends on.

float16 rather than float32 halves the file. Retrieval ranks by cosine distance,
which is invariant to the ~5e-4 relative error that costs; that this changes no
metric is measured in benchmarks/history.md, not assumed.

Storing vectors couples the fixture to the model that produced them, so the file
records which model that was and load refuses to run under a different one.
Without that check, changing the embedding model would load mismatched vectors
and show up as a collapse in recall -- a true failure with a badly misleading
cause.

The fixture is a *sample* of the corpus, not all of it, so that the file stays a
reasonable size and loads quickly. The sample always contains every gold
document -- dropping one would make its question unanswerable and quietly raise
recall -- plus as many distractors as the chunk budget allows. What the budget
excludes is printed, never silent.

    python scripts/fixture.py export                  # budgeted (what CI uses)
    python scripts/fixture.py export --all            # entire corpus
    python scripts/fixture.py export --max-chunks N   # explicit budget
    python scripts/fixture.py load                    # in CI, or to rebuild locally
"""

from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sift.config import settings  # noqa: E402
from sift.db import connect, insert_chunks, upsert_document  # noqa: E402
from sift.embed import embed_passages  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "corpus.jsonl.gz"

# Stored vector precision. Halves the file against float32 for a rounding error
# cosine distance does not care about.
VECTOR_DTYPE = np.float16


def encode_vector(vector) -> str | None:
    """One embedding -> base64 float16, or None if the chunk has no vector.

    psycopg hands back pgvector's own Vector wrapper, which numpy will not coerce
    directly; to_numpy() is the documented way out of it.
    """
    if vector is None:
        return None
    if hasattr(vector, "to_numpy"):
        vector = vector.to_numpy()
    return base64.b64encode(np.asarray(vector, dtype=VECTOR_DTYPE).tobytes()).decode("ascii")


def decode_vector(blob: str | None):
    """Inverse of encode_vector. Returns float32, which is what pgvector wants."""
    if blob is None:
        return None
    return np.frombuffer(base64.b64decode(blob), dtype=VECTOR_DTYPE).astype(np.float32)


def gold_doc_ids() -> set[str]:
    """Every document the eval set depends on. These can never be sampled out."""
    from eval.retrieval_eval import load_eval_set

    gold: set[str] = set()
    for question in load_eval_set():
        gold.update(question.get("gold_doc_ids") or [])
    return gold


def select_documents(conn, max_chunks: int | None) -> tuple[list[str], dict[str, int]]:
    """Choose which documents go in the fixture, under a chunk budget.

    The full corpus is ~55k chunks. The budget used to be about CI time, when
    every chunk was re-embedded on load; now that vectors are committed, the
    binding constraint is the file itself -- roughly 1.1 KB per chunk once the
    text and a float16 vector are gzipped, so the whole corpus would be a ~60 MB
    blob that every clone pays for and every re-export duplicates in history.
    So the fixture is a sample.

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
    """Dump documents, chunk text and chunk vectors to a gzipped JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with connect() as conn:
        selected, stats = select_documents(conn, max_chunks)

        with gzip.open(path, "wt", encoding="utf-8") as fh:
            # First line, so load can reject a mismatched model before spending
            # anything. Written even though nothing but load reads it -- a file
            # of opaque vectors that does not say what produced them is a trap.
            fh.write(
                json.dumps(
                    {
                        "type": "meta",
                        "embedding_model": settings.embedding_model,
                        "vector_dtype": np.dtype(VECTOR_DTYPE).name,
                    }
                )
                + "\n"
            )

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
                       element_types, contains_table, char_count, doc_title,
                       embedding
                FROM chunks WHERE doc_id = ANY(%s) ORDER BY doc_id, chunk_index
                """,
                (selected,),
            ).fetchall()
            unembedded = 0
            for row in rows:
                record = dict(row)
                record["embedding"] = encode_vector(record["embedding"])
                unembedded += record["embedding"] is None
                fh.write(json.dumps({"type": "chunk", **record}) + "\n")

    if unembedded:
        # Load falls back to computing these, so the fixture still works -- but
        # silently shipping a partly-unembedded corpus would make CI slower and
        # its recall worse for a reason nobody would think to look for.
        print(f"  WARNING: {unembedded} chunks had no stored embedding and will be recomputed")

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


def load_fixture(path: Path = FIXTURE, batch_size: int = 256, force: bool = False) -> None:
    """Recreate the corpus in Postgres from the committed documents and vectors.

    This TRUNCATEs first, which is correct in CI -- the database is empty and the
    point is a byte-identical starting state. It is destructive anywhere else,
    and the fixture is a deliberately *smaller* sample than a real ingest, so
    running it against a working corpus silently trades hours of OCR for a
    subset. Refuse when the target holds more documents than the fixture does.
    """
    if not path.exists():
        raise SystemExit(f"No fixture at {path}. Run: python scripts/fixture.py export")

    meta: dict = {}
    documents, chunks = [], []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            kind = record.pop("type")
            if kind == "meta":
                meta = record
            else:
                (documents if kind == "document" else chunks).append(record)

    # Stored vectors are only meaningful under the model that produced them.
    # Queries are embedded live, so loading a fixture built by another model
    # would compare two different vector spaces: retrieval would return noise
    # and the gate would fail for a reason that looks nothing like the cause.
    fixture_model = meta.get("embedding_model")
    if fixture_model and fixture_model != settings.embedding_model:
        raise SystemExit(
            f"Fixture/model mismatch.\n"
            f"  fixture was exported with: {fixture_model}\n"
            f"  config.py currently wants: {settings.embedding_model}\n"
            f"The stored vectors are not comparable to queries embedded by a\n"
            f"different model. Re-export the fixture against a corpus ingested\n"
            f"with the new model: python scripts/fixture.py export"
        )

    with connect() as conn:
        existing = conn.execute(
            "SELECT count(*) AS n FROM documents WHERE parse_status = 'parsed'"
        ).fetchone()["n"]

    if existing > len(documents) and not force:
        raise SystemExit(
            f"Refusing to load: {settings.db_url.rsplit('/', 1)[-1]} already holds "
            f"{existing} parsed documents and this fixture has only {len(documents)}.\n"
            f"Loading would TRUNCATE the larger corpus and replace it with the sample.\n"
            f"  to evaluate the fixture, point SIFT_DB_URL at a scratch database\n"
            f"  to do it anyway, pass --force"
        )

    missing = [c for c in chunks if c.get("embedding") is None]
    print(f"Loading {len(documents)} documents and {len(chunks)} chunks")
    if missing:
        print(f"Embedding {len(missing)} chunks with no stored vector ({settings.embedding_model})")

    with connect() as conn:
        # Worth knowing, and not fixable here: loading the same fixture twice
        # does not build the same HNSW graph. Two clean builds from this exact
        # file disagreed on 3 of 36 questions. `SELECT setseed()` does not help
        # -- it seeds the generator behind random(), which is not the one
        # pgvector draws element levels from. The mitigation is at query time
        # (settings.hnsw_ef_search); see sift/config.py.
        conn.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
        for doc in documents:
            upsert_document(conn, doc)

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            stale = [c for c in batch if c.get("embedding") is None]
            if stale:
                for row, vector in zip(stale, embed_passages([c["text"] for c in stale])):
                    row["embedding"] = vector
            for row in batch:
                if isinstance(row["embedding"], str):
                    row["embedding"] = decode_vector(row["embedding"])
            insert_chunks(conn, batch)
            print(f"  {min(start + batch_size, len(chunks))}/{len(chunks)}")

        # Do the post-bulk-load housekeeping now, deliberately, instead of
        # letting autovacuum do it halfway through the evaluation. CI loads the
        # fixture and immediately starts timing queries, so an autovacuum waking
        # up on a freshly written 14k-row table competes with the very thing
        # being measured. Left to itself it turned roughly one clean build in
        # four into a latency failure at p50 330ms against a 250ms budget, while
        # the other builds sat near 100ms.
        #
        # ANALYZE also gives the planner real statistics before the first query
        # rather than after a few, which is ordinary good practice after any
        # bulk load and costs a couple of seconds here.
        conn.execute("VACUUM ANALYZE chunks")
        conn.execute("VACUUM ANALYZE documents")

        counts = conn.execute(
            "SELECT (SELECT count(*) FROM documents) AS d, (SELECT count(*) FROM chunks) AS c"
        ).fetchone()
    print(f"Loaded: {counts['d']} documents, {counts['c']} chunks")


# Chunk budget for the committed fixture. At 14k chunks the file is ~16 MB and
# loads in about 20 seconds, while still leaving 261 distractor documents in the
# index. Raise it to make the benchmark harder and the repository heavier; lower
# it for the reverse. Re-export and re-measure when you change it -- the
# retrieval numbers move with the corpus, which is the whole point of it.
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
        load_fixture(force="--force" in sys.argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
