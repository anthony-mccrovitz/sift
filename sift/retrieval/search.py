"""Hybrid retrieval: dense vectors and lexical search, fused.

Both halves run as SQL against the same table, which is the main reason to put
vectors in Postgres at all. There is no second service to keep in sync, a chunk
and its embedding and its full-text index cannot drift apart, and metadata
filters compose with both -- "NIST documents from 2024" is a WHERE clause, not
a separate filtered-search API.

On the lexical side we use Postgres full-text search (`ts_rank_cd`) rather than
rank_bm25. Not because BM25 is worse -- it is a better ranking function -- but
because rank_bm25 needs the entire corpus resident in the API process and
rebuilt on every start, and it silently goes stale the moment a document is
ingested. ts_rank_cd is maintained by the database on a generated column. That
tradeoff is worth stating plainly rather than claiming BM25 we do not run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sift.config import settings
from sift.db import connect
from sift.embed import embed_query


@dataclass
class Hit:
    """One retrieved chunk, with everything a citation needs."""

    chunk_id: int
    doc_id: str
    title: str
    source: str
    text: str
    page_start: int | None
    page_end: int | None
    published_year: int | None
    contains_table: bool = False

    # Scores are kept separate so the fused ranking stays explainable -- you can
    # see whether a hit arrived by meaning, by keyword, or by both.
    vector_score: float | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float = 0.0
    retrievers: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        page = self.page_start
        return f"{self.doc_id} p.{page}" if page else self.doc_id


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def _where(filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Build a WHERE fragment shared by both retrievers.

    Only parameterised values -- never string-interpolated user input.
    """
    filters = filters or {}
    clauses = ["d.parse_status = 'parsed'"]
    params: dict[str, Any] = {}

    if filters.get("source"):
        clauses.append("d.source = %(source)s")
        params["source"] = filters["source"]
    if filters.get("year"):
        clauses.append("d.published_year = %(year)s")
        params["year"] = int(filters["year"])
    if filters.get("year_min"):
        clauses.append("d.published_year >= %(year_min)s")
        params["year_min"] = int(filters["year_min"])
    if filters.get("year_max"):
        clauses.append("d.published_year <= %(year_max)s")
        params["year_max"] = int(filters["year_max"])
    if filters.get("tables_only"):
        clauses.append("c.contains_table")

    return " AND ".join(clauses), params


_SELECT = """
    c.id, c.doc_id, c.text, c.page_start, c.page_end, c.contains_table,
    d.title, d.source, d.published_year
"""


def _to_hit(row: dict[str, Any]) -> Hit:
    return Hit(
        chunk_id=row["id"],
        doc_id=row["doc_id"],
        title=row["title"] or row["doc_id"],
        source=row["source"],
        text=row["text"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        published_year=row["published_year"],
        contains_table=bool(row["contains_table"]),
    )


# ---------------------------------------------------------------------------
# dense
# ---------------------------------------------------------------------------

def vector_search(query: str, top_k: int | None = None, filters: dict | None = None) -> list[Hit]:
    """Cosine nearest neighbours over pgvector's HNSW index.

    `<=>` is cosine *distance*, so smaller is better; we flip it to a similarity
    for readability. Embeddings are pre-normalised (see embed.py), which makes
    this exact rather than approximate up to the index's recall.

    The two-stage shape is about determinism, not style. Distance ties are not
    hypothetical here: government PDFs repeat boilerplate headers and footers, so
    byte-identical chunks get byte-identical vectors and therefore exactly equal
    distances. `ORDER BY distance LIMIT k` leaves the order among those ties to
    the executor, which is free to change it between runs -- and it does, which
    made the regression gate return different metrics for identical data.

    The obvious fix, `ORDER BY c.embedding <=> q, c.id`, is wrong: a compound
    sort key the HNSW index cannot serve makes the planner abandon the index for
    a full sequential scan (verified with EXPLAIN). So the inner query keeps the
    plain index-ordered LIMIT, and the outer query imposes a total order on the
    rows it returned.
    """
    top_k = top_k or settings.vector_top_k
    where, params = _where(filters)
    vector = embed_query(query)

    sql = f"""
        SELECT *, 1 - distance AS score FROM (
            SELECT {_SELECT}, c.embedding <=> %(qvec)s AS distance
            FROM chunks c JOIN documents d USING (doc_id)
            WHERE {where} AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %(qvec)s
            LIMIT %(k)s
        ) ranked
        ORDER BY distance, id
    """
    with connect() as conn:
        rows = conn.execute(sql, {**params, "qvec": vector, "k": top_k}).fetchall()

    hits = []
    for rank, row in enumerate(rows, start=1):
        hit = _to_hit(row)
        hit.vector_score = float(row["score"])
        hit.vector_rank = rank
        hit.retrievers = ["vector"]
        hits.append(hit)
    return hits


# ---------------------------------------------------------------------------
# lexical
# ---------------------------------------------------------------------------

# Question scaffolding carries no lexical signal but, ANDed into a tsquery,
# guarantees zero results.
_QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "did", "does", "do", "is", "are", "was", "were", "the", "a", "an", "of",
    "in", "on", "for", "to", "and", "or", "about", "say", "said", "says",
    "tell", "me", "find", "any", "some", "that", "this", "it", "its", "with",
    "from", "by", "as", "at", "be", "been", "have", "has", "had", "can",
    "could", "should", "would", "there", "their", "they",
}

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-./]*")


def build_tsquery(query: str) -> str:
    """Turn a natural-language question into a websearch_to_tsquery string.

    This is the fix for a bug that made lexical search look useless: both
    websearch_to_tsquery and plainto_tsquery join terms with AND, so
    "what weaknesses did GAO find in export controls for missile technology"
    becomes a demand that one 1800-character chunk contain *every* one of those
    words. It reliably returned zero rows, which then silently degraded hybrid
    retrieval to vector-only without anything appearing to be wrong.

    We drop question scaffolding and OR the content terms together. Recall comes
    from the OR; precision comes from ts_rank_cd, which rewards chunks matching
    more terms and matching them close together. Quoted phrases are preserved
    verbatim so exact-phrase search still works.
    """
    phrases = re.findall(r'"([^"]+)"', query)
    remainder = re.sub(r'"[^"]*"', " ", query)

    terms = [
        w for w in _WORD_RE.findall(remainder)
        if len(w) > 1 and w.lower() not in _QUESTION_WORDS
    ]

    parts = [f'"{p}"' for p in phrases] + terms
    if not parts:
        # Everything was a stopword -- fall back to the raw query rather than
        # sending an empty tsquery, which matches nothing.
        parts = _WORD_RE.findall(query) or [query]

    return " OR ".join(parts)


_IDENTIFIER_RE = re.compile(
    r"\b(?:gao|nist|sp|nsiad|rced|hehs|aimd|imtec|ggd|ocg)[-\s./]?\d{1,4}(?:[-\s./]\d{1,6})?[a-z]?\d?\b",
    re.IGNORECASE,
)


def doc_id_patterns(query: str) -> list[str]:
    """Turn document identifiers in a query into ILIKE patterns for doc_id.

    Needed because identifiers live in the *filename-derived* doc_id, and the
    separators never match: a user types "GAO/NSIAD-95-82" while the corpus
    stores "gaoreports-nsiad-95-82". Splitting the identifier on its separators
    and rejoining with wildcards bridges the two:

        GAO/NSIAD-95-82  ->  %gao%nsiad%95%82%   matches gaoreports-nsiad-95-82
        SP 1271          ->  %sp%1271%           matches nist-sp-1271-final

    Without this, a bare identifier query returned nothing at all -- the exact
    lookup a government user is most likely to try first.
    """
    patterns = []
    for ident in _IDENTIFIER_RE.findall(query):
        parts = [p for p in re.split(r"[-\s./]+", ident) if p]
        if parts:
            patterns.append("%" + "%".join(p.lower() for p in parts) + "%")
    return patterns


def keyword_search(query: str, top_k: int | None = None, filters: dict | None = None) -> list[Hit]:
    """Full-text search over the generated tsvector column.

    websearch_to_tsquery is the right parser for user input: it accepts quoted
    phrases and OR/-negation, and -- crucially -- it does not raise a syntax
    error on arbitrary text the way to_tsquery does. Feeding raw user queries to
    to_tsquery is a reliable way to 500 your own API.
    """
    top_k = top_k or settings.keyword_top_k
    where, params = _where(filters)
    tsquery = build_tsquery(query)
    patterns = doc_id_patterns(query)

    # A chunk qualifies on either signal: a full-text match, or belonging to a
    # document whose id matches an identifier in the query. Identifier matches
    # are scored above every text match so an exact lookup wins outright.
    #
    # `, c.id` is the tiebreak, and it matters more here than on the dense side.
    # ts_rank_cd ties constantly -- every chunk matching the same terms the same
    # number of times scores identically, and an identifier lookup gives a whole
    # document the same 100 + epsilon. Without a tiebreak, `LIMIT k` over
    # hundreds of tied rows returns an arbitrary k of them, so the same query
    # against the same corpus can return different documents on different runs.
    # This one is free: the plan already materialises a sort here, so the extra
    # key rides along with it. The dense side could not do the same -- see
    # vector_search.
    sql = f"""
        SELECT {_SELECT},
               CASE WHEN %(patterns)s::text[] IS NOT NULL
                         AND c.doc_id ILIKE ANY(%(patterns)s::text[])
                    THEN 100 + ts_rank_cd(c.tsv, q)
                    ELSE ts_rank_cd(c.tsv, q)
               END AS score
        FROM chunks c
        JOIN documents d USING (doc_id),
             websearch_to_tsquery('english', %(q)s) AS q
        WHERE {where}
          AND (c.tsv @@ q
               OR (%(patterns)s::text[] IS NOT NULL AND c.doc_id ILIKE ANY(%(patterns)s::text[])))
        ORDER BY score DESC, c.id
        LIMIT %(k)s
    """
    with connect() as conn:
        rows = conn.execute(
            sql, {**params, "q": tsquery, "k": top_k, "patterns": patterns or None}
        ).fetchall()

    hits = []
    for rank, row in enumerate(rows, start=1):
        hit = _to_hit(row)
        hit.keyword_score = float(row["score"])
        hit.keyword_rank = rank
        hit.retrievers = ["keyword"]
        hits.append(hit)
    return hits


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    runs: list[list[Hit]], k: int | None = None, top_k: int | None = None
) -> list[Hit]:
    """Combine ranked lists by rank, not by score.

    RRF scores each document as sum(1 / (k + rank)) across the lists it appears
    in. Using ranks is the whole trick: cosine similarity (~0.4-0.8) and
    ts_rank_cd (unbounded, corpus-dependent) are not on comparable scales, and
    any attempt to normalise them into a weighted sum needs constants that are
    wrong as soon as the corpus changes. Ranks are always comparable.

    k=60 is the value from Cormack et al. (2009); it damps the influence of the
    very top ranks so one retriever cannot dominate on its own.
    """
    k = k or settings.rrf_k
    top_k = top_k or settings.final_top_k

    merged: dict[int, Hit] = {}
    for run in runs:
        for rank, hit in enumerate(run, start=1):
            existing = merged.get(hit.chunk_id)
            if existing is None:
                merged[hit.chunk_id] = hit
                existing = hit
            else:
                # Same chunk found by both retrievers: keep both scores so the
                # response can show why it surfaced.
                existing.vector_score = existing.vector_score or hit.vector_score
                existing.keyword_score = existing.keyword_score or hit.keyword_score
                existing.vector_rank = existing.vector_rank or hit.vector_rank
                existing.keyword_rank = existing.keyword_rank or hit.keyword_rank
                for r in hit.retrievers:
                    if r not in existing.retrievers:
                        existing.retrievers.append(r)
            existing.fused_score += 1.0 / (k + rank)

    # Tie-break on chunk_id, not insertion order. Exact score ties are common --
    # two chunks each found at rank 1 by one retriever score identically -- and
    # Python's stable sort would otherwise let the order the runs happened to
    # arrive in decide the ranking. That made MRR shift by 0.02 between runs on
    # an identical corpus purely because reloading the database renumbered rows.
    # A retrieval system whose output depends on insertion order cannot have a
    # meaningful regression gate.
    ranked = sorted(merged.values(), key=lambda h: (-h.fused_score, h.chunk_id))
    return ranked[:top_k]


def hybrid_search(
    query: str,
    top_k: int | None = None,
    filters: dict | None = None,
    use_vector: bool = True,
    use_keyword: bool = True,
) -> list[Hit]:
    runs = []
    if use_vector:
        runs.append(vector_search(query, filters=filters))
    if use_keyword:
        runs.append(keyword_search(query, filters=filters))
    if not runs:
        return []
    if len(runs) == 1:
        return runs[0][: top_k or settings.final_top_k]
    return reciprocal_rank_fusion(runs, top_k=top_k)
