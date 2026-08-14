"""Turn a PDF into Unstructured elements, and survive the ones that fight back.

Three things here are load-bearing, and each exists because of a real failure:

1. **We probe the text layer before choosing a strategy.** Unstructured's
   `strategy="auto"` decides for itself, but on our corpus it is wrong in a
   specific, damaging way: 1960s NASA scans carry a text layer from 1990s-era
   OCR, so `auto` sees text, picks `fast`, and returns confident garbage
   ("UNSTEADYAERODYNAMICS", words in the wrong order). We measure the text
   layer's *quality*, not just its presence.

2. **Every parse runs in a child process with a hard timeout.** hi_res parsing
   calls into native code (onnxruntime, poppler, tesseract). Native code can
   segfault or hang forever, and neither is catchable with try/except in-process.
   One bad PDF must not take down a 500-document run.

3. **Failures are data.** A document that fails is written to `documents` with
   the exception class and message, not dropped. That table is where the
   README's "what broke" section comes from.
"""

from __future__ import annotations

import multiprocessing as mp
import re
import statistics
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# "fast" is pdfminer text extraction: milliseconds per page, requires a good
# text layer. "hi_res" runs layout detection and OCR: seconds per page, but it
# is the only thing that reads a scan. "ocr_only" skips layout detection.
STRATEGY_FAST = "fast"
STRATEGY_HI_RES = "hi_res"
STRATEGY_OCR = "ocr_only"


@dataclass
class ProbeResult:
    """What we learned about a PDF before committing to a parse strategy."""

    page_count: int = 0
    chars_per_page: float = 0.0
    garbled_ratio: float = 0.0  # 0 = clean prose, 1 = OCR soup
    is_scanned: bool = False
    error: str | None = None


@dataclass
class ParseResult:
    elements: list[Any] = field(default_factory=list)
    strategy: str | None = None
    page_count: int = 0
    element_count: int = 0
    is_scanned: bool = False
    seconds: float = 0.0
    status: str = "parsed"  # parsed | failed | empty
    error: str | None = None


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------

_LONE_LETTER = re.compile(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])")


def _sample_page_numbers(page_count: int, k: int = 4) -> list[int]:
    """Pick pages to probe -- from a third of the way in, never the front.

    Front matter lies. Title pages and tables of contents are legitimately one
    word per line, so a document sampled at page 1 looks exactly like a bad
    scan. Sampling the body fixed a false-positive rate that would have sent
    every long NIST publication down the slow OCR path.
    """
    if page_count <= k:
        return list(range(page_count))
    start = max(1, page_count // 3)
    return list(range(start, min(page_count, start + k)))


def _garbled_score(text: str) -> float:
    """0..1 estimate of how unusable an extracted text layer is.

    The signal that works is *layout*, not spelling. A scan run through
    decades-old OCR yields real words in shattered order -- one word per line,
    columns interleaved -- so per-word checks see nothing wrong. We instead ask:
    what fraction of the text lives in lines shaped like prose?

      clean, born-digital PDF : ~0.95
      1967 microfiche scan    : ~0.17
    """
    if len(text) < 200:
        return 1.0

    # Lines of 1-2 characters are rotated/vertical text artifacts, not content.
    # NIST publications carry a sidebar watermark that pdfminer emits one
    # character per line; left in, ~130 junk lines per page swamp the measure.
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 3]
    words = text.split()
    if not lines or not words:
        return 1.0

    total_chars = sum(len(ln) for ln in lines)
    prose_chars = sum(len(ln) for ln in lines if len(ln.split()) >= 4)
    prose_fraction = prose_chars / max(1, total_chars)

    lone_ratio = len(_LONE_LETTER.findall(text)) / len(words)
    median_words = statistics.median(len(ln.split()) for ln in lines)

    score = (1.0 - prose_fraction) * 0.75
    if median_words <= 2:
        score += 0.15
    score += min(0.10, lone_ratio * 0.5)
    return min(1.0, score)


def probe_pdf(path: Path, sample_pages: int = 4) -> ProbeResult:
    """Cheaply inspect a PDF's text layer to decide how to parse it."""
    from pdfminer.high_level import extract_text
    from pypdf import PdfReader

    probe = ProbeResult()
    try:
        reader = PdfReader(str(path))
        probe.page_count = len(reader.pages)
    except Exception as exc:  # noqa: BLE001 -- encrypted/corrupt files land here
        probe.error = f"{type(exc).__name__}: {exc}"
        probe.is_scanned = True  # unknown -> assume the expensive path
        return probe

    pages = _sample_page_numbers(probe.page_count, sample_pages)
    try:
        text = extract_text(str(path), page_numbers=pages) or ""
    except Exception as exc:  # noqa: BLE001
        probe.error = f"{type(exc).__name__}: {exc}"
        probe.is_scanned = True
        return probe

    probe.chars_per_page = len(text) / max(1, len(pages))
    probe.garbled_ratio = _garbled_score(text)

    # Under ~120 chars/page there is effectively no text layer. Above that but
    # garbled means a bad historical OCR pass -- both need real OCR.
    probe.is_scanned = probe.chars_per_page < 120 or probe.garbled_ratio > 0.55
    return probe


def choose_strategy(probe: ProbeResult) -> str:
    """Pick a parse strategy from the probe.

    Scanned documents go to `ocr_only`, NOT `hi_res`, and that is the opposite
    of what the obvious reading of Unstructured's docs suggests. It is a
    measured decision -- see scripts/compare_strategies.py, run against a 1967
    NASA microfiche scan:

        strategy    seconds   garbled   bigrams/1k (readability)
        fast           11.7      0.81        48.5
        hi_res         40.1      0.05        23.9   <- worst to read
        ocr_only       36.3      0.04        50.4   <- best, and faster

    hi_res *looks* the cleanest by any layout measure, because it produces
    well-formed lines. But it runs layout detection first, and on a low-contrast
    microfiche scan it carves the page into regions and emits them in the wrong
    reading order -- correct words, scrambled sentences. The common-bigram score
    catches this; a garbled-layout score cannot, because the layout is fine.
    It is the failure mode that looks like success.

    ocr_only skips region detection and reads the page straight through, which
    on single-column scans is both faster and markedly more faithful.

    The tradeoff we accept: ocr_only does not reconstruct table structure. For
    this corpus that is the right trade, because the scanned documents are
    1960s-70s technical prose. A corpus of scanned *financial tables* should
    revisit it -- force_strategy exists for that.
    """
    if probe.chars_per_page < 40:
        return STRATEGY_OCR      # nothing to work with at all
    if probe.is_scanned:
        return STRATEGY_OCR      # measured better than hi_res on real scans
    return STRATEGY_FAST


# ---------------------------------------------------------------------------
# parsing, isolated in a child process
# ---------------------------------------------------------------------------

def _partition_worker(path: str, strategy: str, queue: mp.Queue) -> None:
    """Runs in a child process. Must only put picklable things on the queue."""
    try:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=path,
            strategy=strategy,
            # Tables are the whole reason to use a layout-aware parser on
            # government PDFs; without this they arrive as scrambled prose.
            infer_table_structure=strategy != STRATEGY_FAST,
            languages=["eng"],
        )
        # Element objects do not pickle reliably across a process boundary, so
        # we use Unstructured's own serialization. It round-trips through
        # elements_from_dicts() with metadata (page numbers, table HTML) intact,
        # which is what the chunker and our citations depend on.
        from unstructured.staging.base import elements_to_dicts

        queue.put({"ok": True, "elements": list(elements_to_dicts(elements))})
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4),
            }
        )


def compute_timeout(strategy: str, page_count: int, base: int = 300) -> int:
    """Scale the parse timeout with the work actually required.

    Measured on this corpus: `fast` runs ~0.2s/page, `hi_res` ~36s/page (layout
    detection plus OCR on every page). A single flat timeout cannot serve both
    -- 300s is generous for a 500-page text PDF and kills a 12-page scan halfway
    through. So the budget follows the strategy and the page count, with a
    ceiling so one pathological document cannot stall a run indefinitely.
    """
    if strategy == STRATEGY_FAST:
        return base
    per_page = 45  # measured 36s/page, plus headroom
    return int(min(3600, max(base, per_page * max(1, page_count))))


def partition_document(
    path: Path, timeout: int | None = None, force_strategy: str | None = None
) -> ParseResult:
    """Parse one PDF. Returns a ParseResult; never raises."""
    result = ParseResult()
    started = time.monotonic()

    if not path.exists():
        result.status, result.error = "failed", "FileNotFound: never downloaded"
        return result

    probe = probe_pdf(path)
    result.page_count = probe.page_count
    result.is_scanned = probe.is_scanned
    strategy = force_strategy or choose_strategy(probe)
    result.strategy = strategy
    timeout = timeout or compute_timeout(strategy, probe.page_count)

    # 'spawn' keeps the child clean of the parent's already-imported native
    # libraries, which otherwise deadlock on fork under macOS.
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_partition_worker, args=(str(path), strategy, queue))
    proc.start()

    try:
        payload = queue.get(timeout=timeout)
    except Exception:  # queue.Empty, or the child died without writing
        proc.terminate()
        proc.join(5)
        result.seconds = time.monotonic() - started
        result.status = "failed"
        if proc.exitcode is not None and proc.exitcode < 0:
            result.error = f"ParserCrash: worker killed by signal {-proc.exitcode} (strategy={strategy})"
        else:
            result.error = f"ParseTimeout: exceeded {timeout}s (strategy={strategy})"
        return result
    finally:
        proc.join(5)
        if proc.is_alive():
            proc.kill()

    result.seconds = time.monotonic() - started

    if not payload.get("ok"):
        result.status = "failed"
        result.error = payload.get("error", "unknown parser error")
        return result

    elements = [e for e in payload["elements"] if (e.get("text") or "").strip()]
    result.elements = elements
    result.element_count = len(elements)

    if not elements:
        # Not a crash, but not a success either. A scanned page that OCR could
        # not read produces exactly this, and it deserves its own status so it
        # shows up in the report instead of looking like a healthy document.
        result.status = "empty"
        result.error = f"NoTextExtracted: 0 usable elements (strategy={strategy})"

    return result
