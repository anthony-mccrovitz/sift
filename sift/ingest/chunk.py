"""Group elements into retrievable passages using Unstructured's chunk_by_title.

Why not a character splitter: a 1800-character window slid over a GAO report
cuts through the middle of tables and separates a heading from the paragraph it
introduces. chunk_by_title starts a new chunk at each section heading, so a
chunk is a *section* -- which is also the unit a human would cite.

The other half of the job is metadata. A citation needs a page number, and page
numbers live on the individual elements, not on the chunk. We recover them from
the chunk's `orig_elements` (which is why include_orig_elements=True below is
not optional).
"""

from __future__ import annotations

from typing import Any

from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_dicts

from sift.config import settings

# Element categories that carry no retrievable meaning. Keeping them pollutes
# chunks with page furniture and, worse, wastes the LLM's context window.
NOISE_CATEGORIES = {"Header", "Footer", "PageBreak", "PageNumber", "UncategorizedText"}


def _page_range(chunk: Any) -> tuple[int | None, int | None]:
    """Smallest and largest page touched by a chunk's source elements."""
    pages: list[int] = []

    orig = getattr(chunk.metadata, "orig_elements", None) or []
    for el in orig:
        page = getattr(el.metadata, "page_number", None)
        if isinstance(page, int):
            pages.append(page)

    if not pages:
        page = getattr(chunk.metadata, "page_number", None)
        if isinstance(page, int):
            pages.append(page)

    return (min(pages), max(pages)) if pages else (None, None)


def _element_types(chunk: Any) -> list[str]:
    orig = getattr(chunk.metadata, "orig_elements", None) or []
    seen = {el.category for el in orig if getattr(el, "category", None)}
    return sorted(seen) or [chunk.category]


def chunk_elements(
    element_dicts: list[dict[str, Any]], doc_id: str, doc_title: str | None = None
) -> list[dict[str, Any]]:
    """Element dicts -> chunk rows ready for embedding and insertion."""
    elements = list(elements_from_dicts(element_dicts))

    # Drop page furniture, but never drop Tables -- they are the point.
    filtered = [
        el
        for el in elements
        if el.category not in NOISE_CATEGORIES and (el.text or "").strip()
    ]
    if not filtered:
        return []

    chunks = chunk_by_title(
        filtered,
        max_characters=settings.chunk_max_chars,
        new_after_n_chars=int(settings.chunk_max_chars * 0.85),
        combine_text_under_n_chars=settings.chunk_combine_under,
        overlap=settings.chunk_overlap,
        # Required for page-number recovery above.
        include_orig_elements=True,
        # A section that runs across a page break is still one section.
        multipage_sections=True,
    )

    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text = (chunk.text or "").strip()
        if len(text) < 40:  # not worth an embedding or a row
            continue

        page_start, page_end = _page_range(chunk)
        types = _element_types(chunk)
        rows.append(
            {
                "doc_id": doc_id,
                "chunk_index": index,
                "text": text,
                "page_start": page_start,
                "page_end": page_end,
                "element_types": types,
                "contains_table": "Table" in types,
                "char_count": len(text),
                # Denormalised so the generated tsvector can weight it -- see
                # the doc_title comment in sql/001_schema.sql.
                "doc_title": doc_title,
            }
        )
    return rows
