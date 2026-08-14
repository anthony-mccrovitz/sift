-- Sift schema.
--
-- Two tables. `documents` is one row per source PDF and is the honest record of
-- what happened to it -- including the ones that failed to parse, which we keep
-- rather than silently drop. `chunks` is one row per retrievable passage.
--
-- The interesting design decision is that a single table carries BOTH the dense
-- vector and the full-text index. Hybrid retrieval is then two queries against
-- one store instead of two systems to keep in sync.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,        -- stable slug, e.g. "gao-24-106175"
    source          TEXT NOT NULL,           -- gao | nist | nasa
    title           TEXT,
    url             TEXT,
    local_path      TEXT,
    published_year  INT,

    -- Ingest telemetry. This is what the README's "what broke" section is built
    -- from, so it is first-class data, not a log file we throw away.
    parse_status    TEXT NOT NULL DEFAULT 'pending',  -- pending|parsed|failed|empty
    parse_strategy  TEXT,                    -- fast | hi_res | ocr_only
    is_scanned      BOOLEAN DEFAULT FALSE,   -- no extractable text layer
    page_count      INT,
    element_count   INT,
    chunk_count     INT,
    parse_seconds   REAL,
    parse_error     TEXT,                    -- exception class + message, if failed

    file_sha256     TEXT,
    ingested_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source);
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (parse_status);
CREATE INDEX IF NOT EXISTS documents_year_idx   ON documents (published_year);

-- ---------------------------------------------------------------------------
-- chunks
-- ---------------------------------------------------------------------------
-- NOTE: vector(384) matches BAAI/bge-small-en-v1.5. If you change the embedding
-- model in config, change this dimension too -- pgvector fixes it at DDL time.
CREATE TABLE IF NOT EXISTS chunks (
    id             BIGSERIAL PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index    INT  NOT NULL,

    text           TEXT NOT NULL,
    -- Citations need a page. Unstructured gives per-element page numbers; a
    -- chunk can span a page break, so we keep the range and cite the start.
    page_start     INT,
    page_end       INT,
    element_types  TEXT[],                   -- {Title,NarrativeText,Table}
    contains_table BOOLEAN DEFAULT FALSE,
    char_count     INT,

    -- Denormalised from documents so the full-text index can cover it. A
    -- generated column cannot reach into another table, and without the title
    -- in the index a question like "what is NIST SP 1271?" cannot find the
    -- document *titled* "SP 1271, Getting Started with the NIST Cybersecurity
    -- Framework" -- the identifier appears in the title and nowhere in the body.
    doc_title      TEXT,

    embedding      vector(384),

    -- Generated column: Postgres keeps the lexical index in sync automatically,
    -- so lexical and vector search cannot drift apart.
    --
    -- setweight gives title matches rank 'A' and body matches rank 'B'.
    -- ts_rank_cd then scores a title hit well above an incidental body
    -- mention, which is what makes identifier lookups land on the right document.
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(doc_title, '')), 'A') ||
        setweight(to_tsvector('english', text), 'B')
    ) STORED,

    UNIQUE (doc_id, chunk_index)
);

-- HNSW over cosine distance. Built on an empty table it costs nothing; pgvector
-- fills it incrementally as rows arrive.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);

-- ---------------------------------------------------------------------------
-- convenience view: the ingest report, straight out of the database
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ingest_report AS
SELECT
    source,
    parse_status,
    count(*)                          AS documents,
    sum(chunk_count)                  AS chunks,
    round(avg(parse_seconds)::numeric, 2) AS avg_parse_seconds,
    count(*) FILTER (WHERE is_scanned) AS scanned
FROM documents
GROUP BY source, parse_status
ORDER BY source, parse_status;
