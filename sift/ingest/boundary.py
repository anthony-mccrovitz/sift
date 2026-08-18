"""Find where a Federal Register document actually starts and stops.

The problem this solves. The Federal Register API hands out a `pdf_url` per
document, but the PDF behind it is a *page extract* from that day's issue. A rule
that begins halfway down a page inherits the top of that page, and one that ends
halfway down carries whatever starts underneath. So "download document X" gives
you the tail of X-1 and the head of X+1 for free.

Measured on this corpus before the fix: 54 of 80 Federal Register documents held
at least one chunk whose text belonged to a different document. For
fr-2026-16687, chunks 0-2 were the end of fr-2026-16630 and chunks 34-36 were the
beginning of an unrelated FAA airworthiness directive; only 3-33 were the
document itself.

Why it is worth fixing rather than tolerating. Every chunk carries its doc_id
into retrieval and out again as a citation. Foreign text under the wrong doc_id
means the system can quote a passage, attribute it to a document and page, and be
pointing at a different document entirely -- a confident, sourced, wrong answer,
which is the single failure this project exists to prevent. It also quietly
corrupts the eval: q017's ground_truth was drafted from a passage that belongs to
fr-2026-16630 while being labelled fr-2026-16687, and the README explained the
resulting miss as a retrieval limitation. It was not.

How the boundary is found. Every Federal Register document ends with its own
filing footer:

    [FR Doc. 2026-16687 Filed 8-13-26; 8:45 am]

That marker is unambiguous and machine-readable. The document's own footer marks
its end; the last *foreign* footer before that marks where the previous document
finished, so ours begins after it. Anything past our own footer belongs to
whatever came next.

Deliberately stdlib-only and operating on plain strings, so it can be tested
without `unstructured` installed -- the same reason preflight.py is its own
module.
"""

from __future__ import annotations

import re
from typing import Sequence

# Matches "[FR Doc. 2026-16687" and the en-dash and em-dash variants the
# typesetting actually uses. The closing bracket and the "Filed ..." tail are
# deliberately not required: OCR and layout extraction break lines in the middle
# of the footer often enough that insisting on the whole thing loses real matches.
FOOTER_RE = re.compile(r"\[FR\s*Doc\.?\s*(\d{4})[-‐-―](\d{3,6})", re.IGNORECASE)


def document_number(doc_id: str) -> str | None:
    """`fr-2026-16687` -> `2026-16687`. None for anything that is not an FR id."""
    if not doc_id.lower().startswith("fr-"):
        return None
    return doc_id[3:]


def footers_in(text: str) -> list[str]:
    """Every filing footer in one element, normalised to `YYYY-NNNNN`."""
    return [f"{year}-{number}" for year, number in FOOTER_RE.findall(text or "")]


def document_span(texts: Sequence[str], doc_id: str) -> tuple[int, int, str]:
    """Half-open range of elements belonging to `doc_id`, plus what was decided.

    Returns `(start, end, reason)`. The reason is carried rather than logged
    because it belongs in the ingest telemetry: a span that could not be
    determined should be visible as a fact about that document, not as an absence
    of a log line nobody reads.

    Conservative by design. When the document's own footer is absent -- it may be
    the last item in the issue, or the footer may have been mangled past
    recognition -- the whole element list is kept. Trimming on a guess would lose
    real content, and losing content silently is worse than keeping a little of
    someone else's.
    """
    number = document_number(doc_id)
    if number is None:
        return 0, len(texts), "not-federal-register"

    own_end: int | None = None
    last_foreign_before: int | None = None

    for index, text in enumerate(texts):
        found = footers_in(text)
        if not found:
            continue
        if number in found:
            # Our own footer. Everything from here on belongs to the next
            # document, so this element is the last one that is ours.
            own_end = index
            break
        last_foreign_before = index

    if own_end is None:
        return 0, len(texts), "own-footer-not-found"

    start = 0 if last_foreign_before is None else last_foreign_before + 1
    end = own_end + 1

    if start >= end:
        # A foreign footer sitting after our own would mean the markers are out
        # of order and the assumption behind this whole module does not hold for
        # this document. Keep everything rather than return nothing.
        return 0, len(texts), "markers-out-of-order"

    trimmed = (start > 0) or (end < len(texts))
    return start, end, "trimmed" if trimmed else "already-clean"
