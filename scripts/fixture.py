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

    python scripts/fixture.py export     # after a full local ingest
    python scripts/fixture.py load       # in CI, or to rebuild a local DB
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


def export_fixture(path: Path = FIXTURE, limit: int | None = None) -> None:
    """Dump documents and chunk text (no vectors) to a gzipped JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with connect() as conn, gzip.open(path, "wt", encoding="utf-8") as fh:
        docs = conn.execute(
            """
            SELECT doc_id, source, title, url, published_year, parse_status,
                   parse_strategy, is_scanned, page_count, element_count,
                   chunk_count, parse_seconds, parse_error
            FROM documents ORDER BY doc_id
            """
        ).fetchall()
        for doc in docs:
            fh.write(json.dumps({"type": "document", **dict(doc)}) + "\n")

        sql = """
            SELECT doc_id, chunk_index, text, page_start, page_end,
                   element_types, contains_table, char_count, doc_title
            FROM chunks ORDER BY doc_id, chunk_index
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
        for row in rows:
            fh.write(json.dumps({"type": "chunk", **dict(row)}) + "\n")

    size_kb = path.stat().st_size / 1024
    print(f"Exported {len(docs)} documents and {len(rows)} chunks -> {path} ({size_kb:.0f} KB)")


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


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"export", "load"}:
        print(__doc__)
        return 1
    if sys.argv[1] == "export":
        export_fixture()
    else:
        load_fixture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
