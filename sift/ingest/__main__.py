"""CLI: python -m sift.ingest [--limit N] [--workers N] [--skip-download]"""

from __future__ import annotations

import argparse
import shutil

from sift.db import healthcheck
from sift.ingest.pipeline import run_ingest

# System binaries pip cannot install. OCR reaches these through pdf2image and
# pytesseract, which shell out and raise per-document when they are missing.
REQUIRED_BINARIES = {
    "pdfinfo": "poppler",
    "pdftoppm": "poppler",
    "tesseract": "tesseract",
}


def preflight() -> list[str]:
    """Fail before parsing if the OCR toolchain is not on PATH.

    Worth doing because of how this breaks otherwise. pdf2image raises
    PDFInfoNotInstalledError per document, and the pipeline is deliberately
    built never to die on one document -- so a missing binary does not stop the
    run. It marks every scanned document 'failed' and continues, and you get a
    completed ingest, a plausible failure log, and a corpus that quietly
    contains none of the scanned documents the project is about.

    That is an environment problem wearing a data problem's clothes. The failure
    log is meant to describe documents, not the machine, so this check keeps
    them separate: missing binaries stop the run before anything is parsed.
    """
    return sorted({pkg for exe, pkg in REQUIRED_BINARIES.items() if not shutil.which(exe)})


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sift.ingest",
        description="Download, parse, chunk, embed and store the Sift corpus.",
    )
    parser.add_argument("--limit", type=int, default=500, help="documents to ingest")
    parser.add_argument("--workers", type=int, default=None, help="parallel parse workers")
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="re-discover documents instead of reusing data/manifest.json",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="parse what is already in data/raw (use when tuning chunking)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip documents already parsed; use to continue an interrupted run",
    )
    args = parser.parse_args()

    missing = preflight()
    if missing:
        print(f"Missing system dependencies: {', '.join(missing)}")
        print("OCR cannot run without them, and every scanned document would be")
        print("recorded as a parse failure rather than as the environment problem")
        print("it actually is. Install them first:")
        print(f"  macOS:  brew install {' '.join(missing)}")
        deb = {"poppler": "poppler-utils", "tesseract": "tesseract-ocr"}
        print(f"  Debian: apt-get install -y {' '.join(deb.get(m, m) for m in missing)}")
        return 1

    health = healthcheck()
    if not health["ok"]:
        print(f"Database unreachable: {health['error']}")
        print("Start it with:  docker compose up -d db")
        return 1

    run_ingest(
        limit=args.limit,
        workers=args.workers,
        refresh_manifest=args.refresh_manifest,
        skip_download=args.skip_download,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
