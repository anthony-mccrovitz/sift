"""Check the OCR toolchain before parsing anything.

Deliberately its own module, importing nothing but the standard library. It sits
below the parsing stack rather than inside it so that checking the environment
never requires loading `unstructured` -- which matters because the unit-test tier
installs neither `unstructured` nor torch, and a test for this check should not
be the thing that drags them in.

Why the check exists at all: pdf2image and pytesseract shell out to system
binaries and raise per document when those are missing. The pipeline is
deliberately built never to die on one document, so a missing binary does not
stop a run. It marks every scanned document 'failed' and carries on, and you get
a completed ingest, a plausible failure log, and a corpus that quietly contains
none of the scanned documents the project is about.

That is an environment problem wearing a data problem's clothes. The failure log
describes documents; it should never be where you find out poppler is missing.
"""

from __future__ import annotations

import shutil

# System binaries pip cannot install, mapped to the package that provides them.
REQUIRED_BINARIES: dict[str, str] = {
    "pdfinfo": "poppler",
    "pdftoppm": "poppler",
    "tesseract": "tesseract",
}

# What to tell people to install, per platform.
APT_PACKAGES: dict[str, str] = {
    "poppler": "poppler-utils",
    "tesseract": "tesseract-ocr",
}


def missing_binaries() -> list[str]:
    """Packages whose binaries are not on PATH. Deduplicated and sorted.

    Reports the package, not the binary that was probed: pdfinfo and pdftoppm
    both ship in poppler, and naming poppler twice is noise in an error message
    someone is reading because something already went wrong.
    """
    return sorted({pkg for exe, pkg in REQUIRED_BINARIES.items() if not shutil.which(exe)})


def format_missing(missing: list[str]) -> str:
    """The message shown when the toolchain is incomplete."""
    apt = " ".join(APT_PACKAGES.get(m, m) for m in missing)
    return (
        f"Missing system dependencies: {', '.join(missing)}\n"
        "OCR cannot run without them, and every scanned document would be\n"
        "recorded as a parse failure rather than as the environment problem it\n"
        "actually is. Install them first:\n"
        f"  macOS:  brew install {' '.join(missing)}\n"
        f"  Debian: apt-get install -y {apt}"
    )
