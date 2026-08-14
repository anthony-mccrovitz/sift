"""Query routing: decide *how* to search before searching.

An honest description of what this is: a rules layer that reads the query,
extracts metadata filters it can prove are there, and picks a retrieval mode.
It is not an LLM agent, it does not plan, and it does not call itself in a
loop. It is called a router because that is what it does.

The reason it exists is concrete. Three queries that need three different
treatments:

    "what did NIST publish in 2024 about zero trust"
        -> a year and a source are stated outright. Searching the whole corpus
           and hoping 2024 NIST documents float to the top is strictly worse
           than a WHERE clause that guarantees it.

    "GAO-24-106175"
        -> an exact identifier. Dense retrieval is actively bad at these: the
           embedding of a document number is close to every other document
           number. Lexical search nails it.

    "how do agencies decide whether a control is effective"
        -> conceptual, no anchors. This is what dense retrieval is for.

Every routing decision is returned in the response so the choice is auditable
rather than mysterious.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Vocabulary the corpus actually uses, mapped to the `source` column.
SOURCE_PATTERNS: dict[str, list[str]] = {
    "nist": [r"\bnist\b", r"\bcsrc\b", r"\bsp\s*800\b", r"\bspecial publication\b"],
    "gao": [r"\bgao\b", r"\bgovernment accountability\b", r"\bgeneral accounting\b"],
    "nasa": [r"\bnasa\b", r"\bntrs\b", r"\baeronautic", r"\blangley\b"],
    "federal_register": [r"\bfederal register\b", r"\bfinal rule\b", r"\brulemaking\b"],
}

# Document identifiers: GAO-24-106175, NIST SP 800-53, SP 800-171r3, NSIAD-95-82
IDENTIFIER_RE = re.compile(
    r"\b(?:gao|nist|sp|nsiad|rced|hehs|imtec|ggd)[-\s.]?\d{1,4}(?:[-\s.]\d{1,6})?[a-z]?\d?\b",
    re.IGNORECASE,
)

YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
# "and" belongs here: "between 1995 and 1999" is the most natural phrasing of a
# range, and without it the two years look like prose and no filter is applied.
YEAR_RANGE_RE = re.compile(
    r"\b(19[5-9]\d|20[0-4]\d)\s*(?:-|–|to|through|until|and)\s*(19[5-9]\d|20[0-4]\d)\b",
    re.IGNORECASE,
)
SINCE_RE = re.compile(r"\b(?:since|after|from)\s+(19[5-9]\d|20[0-4]\d)\b", re.IGNORECASE)
BEFORE_RE = re.compile(r"\b(?:before|prior to|up to|until)\s+(19[5-9]\d|20[0-4]\d)\b", re.IGNORECASE)

TABLE_HINTS = re.compile(
    r"\b(table|tabular|how much|how many|total|figure|percentage|budget|cost|spending|dollar)\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    mode: str  # hybrid | keyword | vector | filtered
    filters: dict[str, Any] = field(default_factory=dict)
    use_vector: bool = True
    use_keyword: bool = True
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"mode={self.mode}"]
        if self.filters:
            parts.append("filters=" + ",".join(f"{k}={v}" for k, v in self.filters.items()))
        return " ".join(parts) + (f" ({'; '.join(self.reasons)})" if self.reasons else "")


def _detect_source(query: str) -> tuple[str | None, str | None]:
    lowered = query.lower()
    for source, patterns in SOURCE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return source, match.group(0).strip()
    return None, None


def _detect_years(query: str) -> tuple[dict[str, int], list[str]]:
    filters: dict[str, int] = {}
    reasons: list[str] = []

    span = YEAR_RANGE_RE.search(query)
    if span:
        lo, hi = sorted((int(span.group(1)), int(span.group(2))))
        filters["year_min"], filters["year_max"] = lo, hi
        reasons.append(f"year range {lo}-{hi}")
        return filters, reasons

    since = SINCE_RE.search(query)
    if since:
        filters["year_min"] = int(since.group(1))
        reasons.append(f"year >= {since.group(1)}")

    before = BEFORE_RE.search(query)
    if before:
        filters["year_max"] = int(before.group(1))
        reasons.append(f"year <= {before.group(1)}")

    if not filters:
        years = YEAR_RE.findall(query)
        # Exactly one bare year is a filter. Two or more is usually prose
        # ("between the 2019 and 2021 reports"), and filtering on one of them
        # would silently discard half the answer.
        if len(set(years)) == 1:
            filters["year"] = int(years[0])
            reasons.append(f"year = {years[0]}")

    return filters, reasons


def route(query: str) -> RoutingDecision:
    """Choose a retrieval strategy for one query. Pure function, easily tested."""
    decision = RoutingDecision(mode="hybrid")
    query = query.strip()

    # --- metadata filters ------------------------------------------------
    source, matched = _detect_source(query)
    if source:
        decision.filters["source"] = source
        decision.reasons.append(f"source={source} (matched {matched!r})")

    year_filters, year_reasons = _detect_years(query)
    decision.filters.update(year_filters)
    decision.reasons.extend(year_reasons)

    if TABLE_HINTS.search(query):
        # A hint, not a filter: restricting to table chunks would wreck recall
        # on "how much did it cost" when the answer is in a sentence.
        decision.reasons.append("quantitative phrasing (tables boosted, not required)")

    # --- mode ------------------------------------------------------------
    identifiers = IDENTIFIER_RE.findall(query)
    if identifiers:
        # Exact-identifier lookup. Dense retrieval hurts here.
        decision.mode = "keyword"
        decision.use_vector = False
        decision.reasons.append(f"document identifier detected: {identifiers[0]!r}")
        return decision

    if len(query.split()) <= 3 and not decision.filters:
        # Very short queries are usually a term, not a question. Lexical search
        # is more precise; dense retrieval on 2 words returns vague neighbours.
        decision.mode = "keyword"
        decision.use_vector = False
        decision.reasons.append("short keyword-style query")
        return decision

    if decision.filters:
        decision.mode = "filtered"
        decision.reasons.append("metadata filter narrows the corpus before ranking")
        return decision

    decision.reasons.append("no metadata anchors; conceptual query")
    return decision
