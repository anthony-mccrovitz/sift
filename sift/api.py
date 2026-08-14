"""FastAPI surface: POST /query, GET /health, GET /documents.

Kept small on purpose. No auth, no sessions, no streaming -- those are product
features, and this is a retrieval system. What it does expose is the *reasoning*
behind each answer: which retrieval mode ran, which filters the router derived,
which passages were used, and what each one scored. An answer you cannot audit
is not much use in a government setting.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from sift.answer import answer_question
from sift.config import settings
from sift.db import connect, healthcheck, ingest_report, list_documents
from sift.llm import llm_available
from sift.retrieval.router import route
from sift.retrieval.search import hybrid_search

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the embedding model before serving traffic.

    sentence-transformers loads lazily, so without this the *first* query pays
    ~9 seconds of model load and every one after it takes ~40ms. That is a
    miserable first impression in a demo and it poisons any latency number
    measured over a short run.
    """
    from sift.embed import embed_query

    embed_query("warmup")
    yield


app = FastAPI(
    title="Sift",
    description="Document intelligence over messy public-sector PDFs.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, examples=["What did NIST publish in 2024 about zero trust?"])
    top_k: int | None = Field(None, ge=1, le=20)
    source: Literal["gao", "nist", "nasa", "federal_register"] | None = None
    year: int | None = Field(None, ge=1950, le=2050)
    use_router: bool = True
    include_passages: bool = Field(False, description="return full passage text, not just citations")


class CitationOut(BaseModel):
    marker: int
    doc_id: str
    title: str
    source: str
    page: int | None
    chunk_id: int


class PassageOut(BaseModel):
    marker: int
    doc_id: str
    title: str
    page: int | None
    text: str
    retrievers: list[str]
    vector_score: float | None
    keyword_score: float | None
    fused_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationOut]
    abstained: bool
    routing: dict[str, Any]
    passages: list[PassageOut] | None = None
    model: str
    latency_ms: int
    retrieval_ms: int
    input_tokens: int
    output_tokens: int
    invalid_citations: list[int]


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    db = healthcheck()
    status = "ok" if db["ok"] and db.get("chunks", 0) > 0 else "degraded"
    return {
        "status": status,
        "database": db,
        "llm": {
            "provider": settings.llm_provider,
            "configured": llm_available(),
            "model": settings.anthropic_model
            if settings.llm_provider == "anthropic"
            else settings.openai_model,
        },
        "embedding_model": settings.embedding_model,
    }


@app.get("/documents")
def documents(
    source: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    with connect() as conn:
        rows = list_documents(conn, source=source, limit=limit, offset=offset)
        report = ingest_report(conn)
    return {"count": len(rows), "documents": rows, "ingest_report": report}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    # Explicit filters from the caller override anything the router inferred.
    filters: dict[str, Any] = {}
    if request.source:
        filters["source"] = request.source
    if request.year:
        filters["year"] = request.year

    result = answer_question(
        request.query,
        top_k=request.top_k,
        filters=filters or None,
        use_router=request.use_router,
    )

    routing = {
        "mode": result.routing.mode if result.routing else "hybrid",
        "filters": result.routing.filters if result.routing else {},
        "reasons": result.routing.reasons if result.routing else [],
        "passages_retrieved": len(result.hits),
    }

    passages = None
    if request.include_passages:
        passages = [
            PassageOut(
                marker=i,
                doc_id=h.doc_id,
                title=h.title,
                page=h.page_start,
                text=h.text,
                retrievers=h.retrievers,
                vector_score=h.vector_score,
                keyword_score=h.keyword_score,
                fused_score=round(h.fused_score, 5),
            )
            for i, h in enumerate(result.hits, start=1)
        ]

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        citations=[CitationOut(**c.to_dict()) for c in result.citations],
        abstained=result.abstained,
        routing=routing,
        passages=passages,
        model=result.model,
        latency_ms=result.latency_ms,
        retrieval_ms=result.retrieval_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        invalid_citations=result.invalid_citations,
    )


@app.get("/search")
def search(
    q: str = Query(..., min_length=2),
    top_k: int = Query(6, ge=1, le=20),
    mode: Literal["hybrid", "vector", "keyword", "auto"] = "auto",
) -> dict[str, Any]:
    """Retrieval without synthesis -- for debugging what the LLM actually saw."""
    decision = route(q) if mode == "auto" else None
    use_vector = mode in {"hybrid", "vector"} or (decision.use_vector if decision else True)
    use_keyword = mode in {"hybrid", "keyword"} or (decision.use_keyword if decision else True)

    hits = hybrid_search(
        q,
        top_k=top_k,
        filters=decision.filters if decision else None,
        use_vector=use_vector,
        use_keyword=use_keyword,
    )
    return {
        "query": q,
        "routing": decision.describe() if decision else mode,
        "results": [
            {
                "doc_id": h.doc_id,
                "title": h.title,
                "page": h.page_start,
                "citation": h.citation,
                "retrievers": h.retrievers,
                "vector_score": h.vector_score,
                "keyword_score": h.keyword_score,
                "fused_score": round(h.fused_score, 5),
                "preview": h.text[:300],
            }
            for h in hits
        ],
    }
