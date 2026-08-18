"""Unit tests for the pure logic -- no database, no network, no model.

Everything here is a function whose behaviour I got wrong at least once while
building, which is a decent definition of what deserves a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sift.answer import _validate_citations  # noqa: E402
from sift.ingest.partition import (  # noqa: E402
    STRATEGY_FAST,
    STRATEGY_OCR,
    ProbeResult,
    _garbled_score,
    _sample_page_numbers,
    choose_strategy,
    compute_timeout,
)
from sift.ingest import preflight  # noqa: E402
from sift.retrieval.router import route  # noqa: E402
from sift.retrieval.search import Hit, build_tsquery, doc_id_patterns, reciprocal_rank_fusion  # noqa: E402


# ---------------------------------------------------------------------------
# lexical query construction
# ---------------------------------------------------------------------------

def test_tsquery_ors_terms_instead_of_anding_them():
    """The bug that made lexical search silently return nothing."""
    q = build_tsquery("what weaknesses did GAO find in export controls")
    assert " OR " in q
    assert "AND" not in q
    # question scaffolding is dropped
    assert "what" not in q.lower().split(" or ")
    assert "GAO" in q and "export" in q


def test_tsquery_preserves_quoted_phrases():
    q = build_tsquery('"safe drinking water act" contamination')
    assert '"safe drinking water act"' in q


def test_tsquery_never_returns_empty_for_stopword_only_input():
    # An empty tsquery matches nothing, so this must degrade gracefully.
    assert build_tsquery("what is the").strip()


@pytest.mark.parametrize(
    "query,doc_id",
    [
        # Note the identifier regex latches onto "NSIAD-95-82" and drops the
        # "GAO/" prefix. That is fine -- what matters is not which substrings
        # end up in the pattern but whether the pattern matches the stored id,
        # so that is what we assert.
        ("GAO/NSIAD-95-82", "gaoreports-nsiad-95-82"),
        ("NIST SP 1271", "nist-sp-1271-final"),
        ("what does SP 800-53 say about access control", "nist-sp-800-53r5-final"),
    ],
)
def test_doc_id_patterns_match_the_stored_id(query, doc_id):
    """The pattern must bridge the separator mismatch between what a user types
    and how the corpus stores the identifier."""
    import re

    patterns = doc_id_patterns(query)
    assert patterns, f"no pattern generated for {query}"
    assert any(
        re.fullmatch(p.replace("%", ".*"), doc_id) for p in patterns
    ), f"{patterns} did not match {doc_id}"


def test_doc_id_patterns_empty_when_no_identifier():
    assert doc_id_patterns("how do agencies assess risk") == []


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def test_router_extracts_source_and_year():
    d = route("what did NIST publish in 2024 about zero trust")
    assert d.filters["source"] == "nist"
    assert d.filters["year"] == 2024
    assert d.mode == "filtered"


def test_router_handles_between_x_and_y_ranges():
    d = route("GAO findings between 1995 and 1999 on export controls")
    assert d.filters["year_min"] == 1995
    assert d.filters["year_max"] == 1999


def test_router_disables_vector_search_for_bare_identifiers():
    """Dense retrieval is actively bad at document numbers."""
    d = route("GAO-24-106175")
    assert d.mode == "keyword"
    assert d.use_vector is False


def test_router_ignores_ambiguous_multiple_years():
    """Two bare years is prose, not a filter -- filtering on one would drop
    half the answer."""
    d = route("compare the 2019 report with the 2021 report")
    assert "year" not in d.filters


def test_router_falls_through_to_hybrid_for_conceptual_queries():
    d = route("how do agencies decide whether a control is effective")
    assert d.mode == "hybrid"
    assert d.use_vector and d.use_keyword


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def _hit(chunk_id: int, retriever: str) -> Hit:
    return Hit(
        chunk_id=chunk_id, doc_id=f"doc-{chunk_id}", title="t", source="gao",
        text="x", page_start=1, page_end=1, published_year=2020,
        retrievers=[retriever],
    )


def test_rrf_rewards_documents_found_by_both_retrievers():
    vector_run = [_hit(1, "vector"), _hit(2, "vector"), _hit(3, "vector")]
    keyword_run = [_hit(3, "keyword"), _hit(4, "keyword")]

    fused = reciprocal_rank_fusion([vector_run, keyword_run], top_k=4)

    # chunk 3 is rank 3 in one run and rank 1 in the other; agreement beats
    # being top of a single list.
    assert fused[0].chunk_id == 3
    assert set(fused[0].retrievers) == {"vector", "keyword"}


def test_rrf_is_order_independent():
    a, b = [_hit(1, "vector"), _hit(2, "vector")], [_hit(2, "keyword"), _hit(1, "keyword")]
    one = [h.chunk_id for h in reciprocal_rank_fusion([a, b], top_k=2)]
    # rebuild, since fusion mutates fused_score on the Hit objects
    a, b = [_hit(1, "vector"), _hit(2, "vector")], [_hit(2, "keyword"), _hit(1, "keyword")]
    two = [h.chunk_id for h in reciprocal_rank_fusion([b, a], top_k=2)]
    assert one == two


# ---------------------------------------------------------------------------
# citation validation
# ---------------------------------------------------------------------------

def test_invalid_citations_are_stripped_not_returned():
    """A model citing [9] when 3 passages were supplied must not keep it."""
    hits = [_hit(i, "vector") for i in (1, 2, 3)]
    text, citations, invalid = _validate_citations(
        "The finding was significant [1]. Another claim [9].", hits
    )
    assert invalid == [9]
    assert "[9]" not in text
    assert [c.marker for c in citations] == [1]


def test_valid_citations_survive():
    hits = [_hit(i, "vector") for i in (1, 2, 3)]
    text, citations, invalid = _validate_citations("A [1] and B [3].", hits)
    assert invalid == []
    assert [c.marker for c in citations] == [1, 3]
    assert "[1]" in text and "[3]" in text


# ---------------------------------------------------------------------------
# parse strategy selection
# ---------------------------------------------------------------------------

def test_garbled_score_flags_shattered_layout():
    """One word per line is the signature of a scan whose reading order died."""
    scrambled = "\n".join(["Unsteady", "aerodynamics", "airflows", "that", "are"] * 60)
    clean = ("The Safe Drinking Water Act, enacted in 1974, requires the "
             "Environmental Protection Agency to establish standards for water systems.\n") * 20
    assert _garbled_score(scrambled) > 0.55
    assert _garbled_score(clean) < 0.35


def test_probe_samples_body_pages_not_front_matter():
    """Title pages and tables of contents look exactly like bad scans."""
    assert _sample_page_numbers(100, k=4) == [33, 34, 35, 36]
    assert _sample_page_numbers(3, k=4) == [0, 1, 2]


def test_scanned_documents_route_to_ocr_only():
    """Measured: ocr_only beats hi_res on real microfiche scans."""
    assert choose_strategy(ProbeResult(chars_per_page=3000, is_scanned=True)) == STRATEGY_OCR
    assert choose_strategy(ProbeResult(chars_per_page=3000, is_scanned=False)) == STRATEGY_FAST
    assert choose_strategy(ProbeResult(chars_per_page=5, is_scanned=True)) == STRATEGY_OCR


def test_timeout_scales_with_pages_for_ocr_but_not_for_fast():
    assert compute_timeout(STRATEGY_FAST, 500) == 300
    assert compute_timeout(STRATEGY_OCR, 40) > 300
    assert compute_timeout(STRATEGY_OCR, 100000) <= 3600  # bounded


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def test_preflight_reports_missing_ocr_binaries(monkeypatch):
    """A missing binary must stop the run, not become 24 'failed' documents.

    The pipeline never dies on a single document, so without this check a
    missing poppler produces a completed ingest whose corpus silently contains
    none of the scanned documents.
    """
    monkeypatch.setattr(preflight.shutil, "which", lambda _exe: None)
    assert preflight.missing_binaries() == ["poppler", "tesseract"]

    monkeypatch.setattr(preflight.shutil, "which", lambda exe: f"/usr/bin/{exe}")
    assert preflight.missing_binaries() == []


def test_preflight_names_the_package_not_the_binary(monkeypatch):
    """pdfinfo and pdftoppm both ship in poppler; reporting it twice is noise."""
    monkeypatch.setattr(
        preflight.shutil, "which", lambda exe: None if exe == "pdfinfo" else f"/usr/bin/{exe}"
    )
    assert preflight.missing_binaries() == ["poppler"]


def test_preflight_message_names_install_commands():
    """The error is read by someone already stuck; it must say what to run."""
    message = preflight.format_missing(["poppler", "tesseract"])
    assert "brew install poppler tesseract" in message
    assert "poppler-utils tesseract-ocr" in message


# ---------------------------------------------------------------------------
# session settings (used by the eval's determinism check)
# ---------------------------------------------------------------------------


def test_session_settings_apply_and_restore():
    """Nesting must not leak: the eval perturbs the planner for one pass only."""
    from sift import db

    assert db._SESSION_GUCS == {}
    with db.session_settings(max_parallel_workers_per_gather="4"):
        assert db._SESSION_GUCS == {"max_parallel_workers_per_gather": "4"}
        with db.session_settings(enable_seqscan="off"):
            assert db._SESSION_GUCS["enable_seqscan"] == "off"
            assert db._SESSION_GUCS["max_parallel_workers_per_gather"] == "4"
        assert "enable_seqscan" not in db._SESSION_GUCS
    assert db._SESSION_GUCS == {}


def test_session_settings_restore_even_when_the_body_raises():
    from sift import db

    with pytest.raises(RuntimeError):
        with db.session_settings(enable_seqscan="off"):
            raise RuntimeError("boom")
    assert db._SESSION_GUCS == {}


def test_session_settings_rejects_a_non_identifier_name():
    """set_config parameterises the value but not the name."""
    from sift import db

    with pytest.raises(ValueError):
        with db.session_settings(**{"enable_seqscan; DROP TABLE chunks": "off"}):
            pass
    assert db._SESSION_GUCS == {}


def test_preflight_module_imports_without_the_parsing_stack():
    """Regression: this check must not drag in unstructured.

    The unit-test tier installs neither unstructured nor torch. Importing
    preflight through sift.ingest.__main__ pulled in the whole parsing stack and
    turned a dependency-free check into a test that could not run in CI.
    """
    assert "unstructured" not in sys.modules


# ---------------------------------------------------------------------------
# Federal Register document boundaries
# ---------------------------------------------------------------------------


def _fr_elements():
    """A page extract shaped like the real thing: previous doc, ours, next doc."""
    return [
        "in their place, 'Geospatial-Intelligence'.",                 # 0 prev doc
        "In 10.1, remove 'General Accounting' and add 'Government "
        "Accountability'. [FR Doc. 2026-16630 Filed 8-13-26; 8:45 am]",  # 1 prev ends
        "Applicability date: This rule is effective September 14.",   # 2 ours starts
        "II. Background HDP is a pay differential authorized by...",  # 3
        "(3) Equipment installation. [FR Doc. 2026-16687 Filed "
        "8-13-26; 8:45 am] BILLING CODE 6325-39-P",                   # 4 ours ends
        "DEPARTMENT OF TRANSPORTATION Federal Aviation Admin",        # 5 next doc
    ]


def test_span_drops_the_previous_and_next_documents():
    from sift.ingest.boundary import document_span

    start, end, reason = document_span(_fr_elements(), "fr-2026-16687")
    assert (start, end) == (2, 5)
    assert reason == "trimmed"


def test_span_keeps_everything_when_the_footer_is_absent():
    """11 of 80 real documents have no own footer. Guessing would lose content."""
    from sift.ingest.boundary import document_span

    texts = ["some rule text", "more rule text"]
    start, end, reason = document_span(texts, "fr-2026-16120")
    assert (start, end) == (0, 2)
    assert reason == "own-footer-not-found"


def test_span_leaves_non_federal_register_documents_alone():
    from sift.ingest.boundary import document_span

    texts = ["GAO found that agencies did not...", "[FR Doc. 2026-16630 Filed]"]
    assert document_span(texts, "gaoreports-nsiad-95-82") == (0, 2, "not-federal-register")


def test_footer_regex_survives_en_dash_typesetting():
    """The Federal Register types these with en-dashes, not hyphens."""
    from sift.ingest.boundary import footers_in

    assert footers_in("[FR Doc. 2026–16687 Filed 8–13–26; 8:45 am]") == ["2026-16687"]
    assert footers_in("[FR Doc. 2026-16687 Filed]") == ["2026-16687"]


def test_the_q017_passage_is_attributed_to_the_document_that_filed_it():
    """Regression for the bug that made q017 look like a retrieval failure.

    The passage answering q017 sits in the *previous* document's span. Before
    the trim it was stored under fr-2026-16687 and the README explained the
    resulting miss as a limit of dense retrieval. It was a mislabelled chunk.
    """
    from sift.ingest.boundary import document_span

    texts = _fr_elements()
    start, end, _ = document_span(texts, "fr-2026-16687")
    kept = " ".join(texts[start:end])
    assert "General Accounting" not in kept
