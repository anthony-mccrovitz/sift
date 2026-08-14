"""Database access. Thin deliberate wrapper over psycopg3 -- no ORM.

An ORM would hide exactly the thing this project is about: the SQL that does
hybrid retrieval. Everything here is a plain function that takes a connection.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterable, Iterator, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from sift.config import settings


@contextlib.contextmanager
def connect(autocommit: bool = True) -> Iterator[psycopg.Connection]:
    """Yield a connection with pgvector's type adapters installed.

    register_vector() is what lets us pass a plain Python list where the schema
    wants vector(384), and get one back on SELECT. Forget it and psycopg will
    stringify your embedding into something Postgres rejects.
    """
    with psycopg.connect(settings.db_url, autocommit=autocommit, row_factory=dict_row) as conn:
        register_vector(conn)
        yield conn


def healthcheck() -> dict[str, Any]:
    """Used by GET /health and by scripts that want to fail fast with a clear
    message instead of a psycopg traceback."""
    try:
        with connect() as conn:
            row = conn.execute(
                """
                SELECT
                  (SELECT count(*) FROM documents)                              AS documents,
                  (SELECT count(*) FROM documents WHERE parse_status = 'parsed') AS parsed,
                  (SELECT count(*) FROM documents WHERE parse_status = 'failed') AS failed,
                  (SELECT count(*) FROM chunks)                                 AS chunks,
                  (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL)     AS embedded
                """
            ).fetchone()
        return {"ok": True, **row}
    except Exception as exc:  # noqa: BLE001 -- health must never raise
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

def upsert_document(conn: psycopg.Connection, doc: dict[str, Any]) -> None:
    """Insert or update the registry row for one source PDF.

    Called twice per document: once at download time (status 'pending') and
    again after parsing with the real telemetry. ON CONFLICT with COALESCE means
    the second call never clobbers a field it does not know about.
    """
    conn.execute(
        """
        INSERT INTO documents (
            doc_id, source, title, url, local_path, published_year,
            parse_status, parse_strategy, is_scanned, page_count,
            element_count, chunk_count, parse_seconds, parse_error, file_sha256
        ) VALUES (
            %(doc_id)s, %(source)s, %(title)s, %(url)s, %(local_path)s, %(published_year)s,
            %(parse_status)s, %(parse_strategy)s, %(is_scanned)s, %(page_count)s,
            %(element_count)s, %(chunk_count)s, %(parse_seconds)s, %(parse_error)s, %(file_sha256)s
        )
        ON CONFLICT (doc_id) DO UPDATE SET
            title          = COALESCE(EXCLUDED.title,          documents.title),
            url            = COALESCE(EXCLUDED.url,            documents.url),
            local_path     = COALESCE(EXCLUDED.local_path,     documents.local_path),
            published_year = COALESCE(EXCLUDED.published_year, documents.published_year),
            parse_status   = EXCLUDED.parse_status,
            parse_strategy = COALESCE(EXCLUDED.parse_strategy, documents.parse_strategy),
            is_scanned     = COALESCE(EXCLUDED.is_scanned,     documents.is_scanned),
            page_count     = COALESCE(EXCLUDED.page_count,     documents.page_count),
            element_count  = COALESCE(EXCLUDED.element_count,  documents.element_count),
            chunk_count    = COALESCE(EXCLUDED.chunk_count,    documents.chunk_count),
            parse_seconds  = COALESCE(EXCLUDED.parse_seconds,  documents.parse_seconds),
            parse_error    = EXCLUDED.parse_error,
            file_sha256    = COALESCE(EXCLUDED.file_sha256,    documents.file_sha256),
            ingested_at    = now()
        """,
        {
            "parse_status": "pending",
            "parse_strategy": None,
            "is_scanned": None,
            "page_count": None,
            "element_count": None,
            "chunk_count": None,
            "parse_seconds": None,
            "parse_error": None,
            "file_sha256": None,
            "title": None,
            "url": None,
            "local_path": None,
            "published_year": None,
            **doc,
        },
    )


def document_status(conn: psycopg.Connection, doc_id: str) -> str | None:
    row = conn.execute(
        "SELECT parse_status FROM documents WHERE doc_id = %s", (doc_id,)
    ).fetchone()
    return row["parse_status"] if row else None


def list_documents(
    conn: psycopg.Connection, source: str | None = None, limit: int = 100, offset: int = 0
) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT doc_id, source, title, url, published_year, parse_status,
               parse_strategy, is_scanned, page_count, chunk_count, parse_error
        FROM documents
        WHERE (%s::text IS NULL OR source = %s)
        ORDER BY source, doc_id
        LIMIT %s OFFSET %s
        """,
        (source, source, limit, offset),
    ).fetchall()


def ingest_report(conn: psycopg.Connection) -> list[dict[str, Any]]:
    return conn.execute("SELECT * FROM ingest_report").fetchall()


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------

def delete_chunks(conn: psycopg.Connection, doc_id: str) -> None:
    """Make re-ingesting a document idempotent."""
    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))


def insert_chunks(conn: psycopg.Connection, rows: Sequence[dict[str, Any]]) -> int:
    """Bulk-insert chunks with their embeddings.

    executemany on psycopg3 pipelines under the hood, which is roughly an order
    of magnitude faster than a loop of execute() for a few hundred rows.
    """
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (
                doc_id, chunk_index, text, page_start, page_end,
                element_types, contains_table, char_count, doc_title, embedding
            ) VALUES (
                %(doc_id)s, %(chunk_index)s, %(text)s, %(page_start)s, %(page_end)s,
                %(element_types)s, %(contains_table)s, %(char_count)s, %(doc_title)s, %(embedding)s
            )
            ON CONFLICT (doc_id, chunk_index) DO NOTHING
            """,
            rows,
        )
    return len(rows)


def count_chunks(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]


def iter_all_chunks(conn: psycopg.Connection, batch: int = 2000) -> Iterable[dict[str, Any]]:
    """Stream every chunk (id + text + citation metadata) without loading the
    whole corpus into memory. Used to build the in-process BM25 index."""
    with conn.cursor(name="chunk_stream") as cur:  # server-side cursor
        cur.itersize = batch
        cur.execute(
            """
            SELECT c.id, c.doc_id, c.chunk_index, c.text, c.page_start, c.page_end,
                   d.title, d.source, d.published_year
            FROM chunks c JOIN documents d USING (doc_id)
            ORDER BY c.id
            """
        )
        yield from cur
