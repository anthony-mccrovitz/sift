"""End-to-end ingest: discover -> download -> parse -> chunk -> embed -> store.

Concurrency shape, which took a couple of tries to get right:

  * Parsing runs in a thread pool, but each thread's real work happens in a
    *child process* (see partition.py). Threads are only there to wait on those
    children, so the GIL is irrelevant and a hung PDF blocks one slot, not the run.
  * Embedding happens on the main thread, one document at a time, as parse
    results arrive. torch already saturates the CPU internally; calling it from
    four threads at once makes it slower, not faster, and risks the model being
    loaded four times.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sift.config import settings
from sift.db import connect, delete_chunks, insert_chunks, ingest_report, upsert_document
from sift.embed import embed_passages
from sift.ingest.chunk import chunk_elements
from sift.ingest.download import download_corpus
from sift.ingest.partition import partition_document
from sift.sources import SourceDoc, build_manifest

MANIFEST_PATH = Path("data/manifest.json")


# ---------------------------------------------------------------------------
# manifest persistence -- so a re-run uses the same corpus, not a new one
# ---------------------------------------------------------------------------

def save_manifest(docs: list[SourceDoc], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(d) for d in docs], indent=2))


def load_manifest(path: Path = MANIFEST_PATH) -> list[SourceDoc]:
    if not path.exists():
        return []
    return [SourceDoc(**item) for item in json.loads(path.read_text())]


# ---------------------------------------------------------------------------
# per-document work
# ---------------------------------------------------------------------------

def parse_and_chunk(doc: SourceDoc) -> dict[str, Any]:
    """Parse one PDF and cut it into chunks. Runs in a worker thread."""
    path = Path(settings.raw_dir) / doc.source / f"{doc.doc_id}.pdf"
    # timeout=None lets partition_document scale the budget to the strategy and
    # page count; settings.parse_timeout_seconds is only the floor.
    parsed = partition_document(path, timeout=None)

    record: dict[str, Any] = {
        "doc_id": doc.doc_id,
        "source": doc.source,
        "parse_status": parsed.status,
        "parse_strategy": parsed.strategy,
        "is_scanned": parsed.is_scanned,
        "page_count": parsed.page_count,
        "element_count": parsed.element_count,
        "parse_seconds": round(parsed.seconds, 2),
        "parse_error": parsed.error,
    }

    if parsed.status != "parsed":
        record["chunk_count"] = 0
        return {"record": record, "chunks": []}

    try:
        chunks = chunk_elements(parsed.elements, doc.doc_id, doc_title=doc.title)
    except Exception as exc:  # noqa: BLE001 -- a chunker crash is still a doc failure
        record["parse_status"] = "failed"
        record["parse_error"] = f"ChunkError: {type(exc).__name__}: {exc}"
        record["chunk_count"] = 0
        return {"record": record, "chunks": []}

    if not chunks:
        record["parse_status"] = "empty"
        record["parse_error"] = "NoChunks: elements parsed but none survived filtering"

    record["chunk_count"] = len(chunks)
    return {"record": record, "chunks": chunks}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run_ingest(
    limit: int = 500,
    workers: int | None = None,
    refresh_manifest: bool = False,
    skip_download: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    workers = workers or settings.ingest_workers
    started = time.monotonic()

    # --- 1. manifest ------------------------------------------------------
    docs = [] if refresh_manifest else load_manifest()
    if len(docs) < limit:
        print(f"Discovering documents (target {limit})...")
        docs = build_manifest(limit)
        save_manifest(docs)
    docs = docs[:limit]
    print(f"Manifest: {len(docs)} documents\n")

    # --- 2. download ------------------------------------------------------
    if not skip_download:
        print("Downloading...")
        tally = download_corpus(docs, workers=max(8, workers))
        print(f"Download: {tally}\n")

    # --- 3. parse + chunk + embed + store --------------------------------
    # Documents that failed to download are already recorded with a precise
    # reason ("no PDF rendition published"). Re-parsing them would only
    # overwrite that with a useless "FileNotFound", so we skip them and keep
    # the diagnosis that actually explains what happened.
    with connect() as conn:
        already_failed = {
            row["doc_id"]
            for row in conn.execute(
                "SELECT doc_id FROM documents WHERE parse_status = 'failed'"
                " AND parse_error LIKE 'download:%'"
            ).fetchall()
        }

        # --resume: skip documents already parsed in an earlier run.
        #
        # A full ingest of this corpus takes hours, most of it OCR, and anything
        # that interrupts it -- a laptop sleeping, the database container
        # stopping -- otherwise costs the whole run. Parse results are already
        # durable in Postgres, so the only thing missing was permission to trust
        # them.
        #
        # 'empty' counts as done: it means the document parsed but yielded no
        # usable text, which re-running the identical parse will not change.
        # Genuine failures are NOT skipped -- those are worth retrying, since
        # they include transient problems.
        #
        # This is opt-in rather than the default because after changing chunking
        # or the parse strategy you want every document reprocessed, and a
        # resume that silently kept stale chunks would be the worse bug.
        done: set[str] = set()
        if resume:
            done = {
                row["doc_id"]
                for row in conn.execute(
                    "SELECT doc_id FROM documents WHERE parse_status IN ('parsed', 'empty')"
                ).fetchall()
            }

    skip = already_failed | done
    to_parse = [d for d in docs if d.doc_id not in skip]
    if already_failed:
        print(f"Skipping {len(already_failed)} documents that failed to download.")
    if done:
        print(f"Resuming: {len(done)} documents already parsed, {len(to_parse)} to go.")
    print()

    print(f"Parsing and embedding ({workers} workers)...")
    stats = {"parsed": 0, "failed": len(already_failed), "empty": 0, "chunks": 0}
    failures: list[dict[str, Any]] = []

    with connect() as conn:
        # Carry the download failures into the report unchanged.
        for row in conn.execute(
            "SELECT doc_id, source, parse_status, parse_strategy, parse_error"
            " FROM documents WHERE doc_id = ANY(%s)",
            (list(already_failed),),
        ).fetchall():
            failures.append(dict(row))

        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(parse_and_chunk, d): d for d in to_parse}

            for i, job in enumerate(futures.as_completed(jobs), start=1):
                doc = jobs[job]
                try:
                    outcome = job.result()
                except Exception as exc:  # noqa: BLE001 -- must never kill the run
                    outcome = {
                        "record": {
                            "doc_id": doc.doc_id,
                            "source": doc.source,
                            "parse_status": "failed",
                            "parse_error": f"WorkerError: {type(exc).__name__}: {exc}",
                            "chunk_count": 0,
                        },
                        "chunks": [],
                    }

                record, chunks = outcome["record"], outcome["chunks"]

                if chunks:
                    # Embedding on the main thread, as results arrive.
                    vectors = embed_passages([c["text"] for c in chunks])
                    for row, vector in zip(chunks, vectors):
                        row["embedding"] = vector
                    delete_chunks(conn, doc.doc_id)  # make re-ingest idempotent
                    insert_chunks(conn, chunks)
                    stats["chunks"] += len(chunks)

                upsert_document(conn, record)
                status = record["parse_status"]
                stats[status] = stats.get(status, 0) + 1
                if status != "parsed":
                    failures.append(record)

                flag = {"parsed": "  ", "empty": "! ", "failed": "X "}.get(status, "  ")
                print(
                    f"{flag}[{i}/{len(jobs)}] {doc.doc_id[:44]:44} "
                    f"{record.get('parse_strategy') or '-':7} "
                    f"{record.get('page_count') or 0:>4}p "
                    f"{record.get('chunk_count', 0):>4}c "
                    f"{record.get('parse_seconds') or 0:>6.1f}s"
                    + (f"  {record['parse_error'][:60]}" if record.get("parse_error") else "")
                )

    # --- 4. the failure log is a deliverable, not a side effect ----------
    # Sourced from the database rather than from this run's results, because a
    # resumed run only sees the documents it processed. Reading the corpus back
    # out means the log describes the corpus as it now stands, which is what the
    # README claims it is, regardless of how many runs it took to build.
    with connect() as conn:
        corpus_failures = [
            dict(row)
            for row in conn.execute(
                "SELECT doc_id, source, parse_status, parse_strategy, parse_error"
                " FROM documents WHERE parse_status NOT IN ('parsed')"
                " ORDER BY parse_status, doc_id"
            ).fetchall()
        ]
        totals = {
            row["parse_status"]: row["n"]
            for row in conn.execute(
                "SELECT parse_status, count(*) AS n FROM documents GROUP BY parse_status"
            ).fetchall()
        }
        total_chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]

    write_failure_log(corpus_failures)

    elapsed = time.monotonic() - started
    print(f"\n{'='*74}")
    print(f"Ingest finished in {elapsed/60:.1f} min")
    print(f"  this run : {stats['parsed']} parsed, {stats['empty']} empty, "
          f"{stats['failed']} failed, {stats['chunks']} chunks")
    print(f"  corpus   : {totals.get('parsed', 0)} parsed, {totals.get('empty', 0)} empty, "
          f"{totals.get('failed', 0)} failed, {totals.get('pending', 0)} pending, "
          f"{total_chunks} chunks")
    if corpus_failures:
        print(f"\n  {settings.failed_log} lists all {len(corpus_failures)}, with reasons.")
    print("=" * 74)

    with connect() as conn:
        for row in ingest_report(conn):
            print(
                f"  {row['source']:>17} {row['parse_status']:>7}: "
                f"{row['documents']:>4} docs, {row['chunks'] or 0:>5} chunks, "
                f"avg {row['avg_parse_seconds'] or 0:>6}s, {row['scanned'] or 0} scanned"
            )

    return {"stats": stats, "seconds": elapsed, "failures": corpus_failures}


def write_failure_log(failures: list[dict[str, Any]]) -> None:
    """A grouped, human-readable record of everything that did not work.

    This file is referenced from the README. It is the honest half of the
    project -- 'we parsed 94%' means nothing without 'and here is the 6%'.
    """
    path = Path(settings.failed_log)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not failures:
        path.write_text("No failures.\n")
        return

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for f in failures:
        kind = (f.get("parse_error") or "Unknown").split(":")[0]
        by_kind.setdefault(kind, []).append(f)

    lines = [
        "# Documents that did not ingest cleanly",
        f"# {len(failures)} documents, grouped by failure kind, most common first.",
        "",
    ]
    for kind, group in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"## {kind}  ({len(group)} documents)")
        for f in group:
            lines.append(
                f"  {f['doc_id']:50} [{f.get('parse_strategy') or '-'}] "
                f"{f.get('parse_error') or ''}"
            )
        lines.append("")
    path.write_text("\n".join(lines))
