"""Answer synthesis with inline citations.

The requirement that shapes everything here: every factual sentence must carry a
citation to a specific document and page, and the system must be willing to say
it does not know. In a government-document setting a confident unsourced answer
is worse than no answer, because it is indistinguishable from a sourced one.

Two mechanisms enforce that, and neither is "we asked the model nicely":

  1. Context passages are numbered, and the model is told to cite by number.
     Numbers are a closed vocabulary -- there is no plausible-looking [12] to
     invent when only 6 passages were supplied.
  2. Every citation the model emits is validated against the passages actually
     retrieved. Anything out of range is dropped before the answer is returned,
     and recorded. A model that cites [9] out of 6 does not get to keep it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from sift.config import settings
from sift.llm import Completion, LLMNotConfigured, get_llm
from sift.retrieval.router import RoutingDecision, route
from sift.retrieval.search import Hit, hybrid_search

SYSTEM_PROMPT = """\
You answer questions about US government and public-sector documents using ONLY \
the numbered source passages provided.

Rules, in priority order:
1. Every factual claim must end with a citation like [1] or [2][3]. Cite the \
passage that actually supports the claim.
2. Use only the passages given. Never use outside knowledge, and never infer \
details that are not written down.
3. If the passages do not answer the question, say exactly what is missing. \
Begin such an answer with "The provided documents do not contain". A short \
honest non-answer is correct and expected; a plausible guess is a failure.
4. Quote figures, dates and identifiers exactly as they appear. Do not round, \
convert, or tidy them.
5. Be concise. Two or three sentences is usually right. Do not restate the \
question or describe your process.

Note: some passages come from OCR of scanned documents and may contain \
mangled words or broken line order. Use them if the meaning is clear; ignore \
a passage that is too corrupted to read rather than guessing at it."""


@dataclass
class Citation:
    marker: int
    doc_id: str
    title: str
    source: str
    page: int | None
    chunk_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }


@dataclass
class Answer:
    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    routing: RoutingDecision | None = None
    model: str = ""
    latency_ms: int = 0
    retrieval_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Populated when the model cited a passage number that did not exist.
    invalid_citations: list[int] = field(default_factory=list)
    abstained: bool = False


CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def build_context(hits: list[Hit]) -> str:
    """Render retrieved passages as a numbered block.

    The header on each passage is what lets the model attribute correctly, and
    what lets a human check the answer without opening the database.
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page = f"page {hit.page_start}" if hit.page_start else "page unknown"
        year = hit.published_year or "n.d."
        blocks.append(
            f"[{i}] {hit.title}\n"
            f"    source: {hit.source.upper()} | document: {hit.doc_id} | {page} | {year}\n"
            f"---\n{hit.text.strip()}\n"
        )
    return "\n".join(blocks)


def _validate_citations(text: str, hits: list[Hit]) -> tuple[str, list[Citation], list[int]]:
    """Keep only citations that point at a passage we actually retrieved."""
    used: dict[int, Citation] = {}
    invalid: list[int] = []

    for match in CITATION_RE.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= len(hits):
            hit = hits[n - 1]
            used.setdefault(
                n,
                Citation(
                    marker=n,
                    doc_id=hit.doc_id,
                    title=hit.title,
                    source=hit.source,
                    page=hit.page_start,
                    chunk_id=hit.chunk_id,
                ),
            )
        elif n not in invalid:
            invalid.append(n)

    # Strip hallucinated markers rather than return a citation to nothing.
    if invalid:
        text = CITATION_RE.sub(
            lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(hits) else "", text
        )

    return text.strip(), [used[k] for k in sorted(used)], invalid


def answer_question(
    query: str,
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
    use_router: bool = True,
) -> Answer:
    started = time.monotonic()

    # --- route + retrieve -------------------------------------------------
    decision = route(query) if use_router else RoutingDecision(mode="hybrid")
    merged_filters = {**decision.filters, **(filters or {})}

    retrieval_start = time.monotonic()
    hits = hybrid_search(
        query,
        top_k=top_k or settings.final_top_k,
        filters=merged_filters,
        use_vector=decision.use_vector,
        use_keyword=decision.use_keyword,
    )

    # A metadata filter can be too strict -- "NIST 2024" finds nothing if the
    # corpus has no 2024 NIST documents. Retrying unfiltered beats an empty
    # answer, and we record that we did it.
    if not hits and merged_filters:
        decision.reasons.append("filtered search returned nothing; retried without filters")
        hits = hybrid_search(query, top_k=top_k or settings.final_top_k)
    retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)

    result = Answer(query=query, answer="", hits=hits, routing=decision, retrieval_ms=retrieval_ms)

    if not hits:
        result.answer = "The provided documents do not contain anything relevant to this question."
        result.abstained = True
        result.latency_ms = int((time.monotonic() - started) * 1000)
        return result

    # --- synthesise -------------------------------------------------------
    user_prompt = f"Source passages:\n\n{build_context(hits)}\n\nQuestion: {query}\n\nAnswer:"

    try:
        client = get_llm()
        completion: Completion = client.complete(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    except LLMNotConfigured as exc:
        # Retrieval still worked; say so instead of pretending the whole
        # system is down. This is what you want when demoing without a key.
        result.answer = (
            f"Retrieval succeeded ({len(hits)} passages) but no LLM is configured: {exc}. "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable synthesis."
        )
        result.latency_ms = int((time.monotonic() - started) * 1000)
        return result

    text, citations, invalid = _validate_citations(completion.text, hits)

    result.answer = text
    result.citations = citations
    result.invalid_citations = invalid
    result.model = completion.model
    result.input_tokens = completion.input_tokens
    result.output_tokens = completion.output_tokens
    result.abstained = text.lower().startswith("the provided documents do not contain")
    result.latency_ms = int((time.monotonic() - started) * 1000)
    return result
