"""Fetch the manifest's PDFs to data/raw/ and register them in Postgres.

Download and parse are separate stages on purpose. Parsing 500 PDFs takes tens
of minutes and you will want to re-run it many times as you tune chunking; you
should not re-download the corpus every time you do.
"""

from __future__ import annotations

import concurrent.futures as futures
import hashlib
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sift.config import settings
from sift.db import connect, upsert_document
from sift.sources import BROWSER_HEADERS, SourceDoc


class NotAPdf(Exception):
    """The server returned 200 and something that is not a PDF.

    Common and worth naming: govinfo serves a full "Page Not Found" HTML page
    with status 200 for packages that have no PDF rendition (some 1990s GAO
    reports exist only as text). Rate limiters do the same with a "slow down"
    page. A downloader that trusts the status code fills your corpus with HTML.

    This is deliberately NOT retried -- a missing rendition is permanent, and
    retrying it three times with backoff wasted ~20 seconds per document for
    a result that could never change.
    """


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(httpx.HTTPError),  # transport errors only
    reraise=True,
)
def _fetch(client: httpx.Client, url: str, dest: Path) -> tuple[int, str]:
    """Download one PDF, verifying it really is one. Returns (bytes, sha256)."""
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        head = b""
        hasher = hashlib.sha256()
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as fh:
            for block in resp.iter_bytes(chunk_size=65536):
                if len(head) < 5:
                    head += block[:5]
                hasher.update(block)
                fh.write(block)

        # The magic number is the only trustworthy signal here.
        if not head.startswith(b"%PDF"):
            body = tmp.read_bytes()[:4000].decode("utf-8", "ignore")
            tmp.unlink(missing_ok=True)
            ctype = resp.headers.get("content-type", "?")
            # Distinguish "this document has no PDF" from "we got throttled",
            # because only one of them is worth coming back for.
            if "Page Not Found" in body or "page not found" in body.lower():
                raise NotAPdf("no PDF rendition published for this package (soft 404)")
            raise NotAPdf(f"expected PDF, got content-type={ctype}")

        tmp.rename(dest)
        return dest.stat().st_size, hasher.hexdigest()


def download_one(doc: SourceDoc, client: httpx.Client, force: bool = False) -> dict:
    """Download a single document and record the outcome. Never raises."""
    out_dir = settings.raw_dir / doc.source
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{doc.doc_id}.pdf"

    row = {
        "doc_id": doc.doc_id,
        "source": doc.source,
        "title": doc.title,
        "url": doc.url,
        "local_path": str(dest),
        "published_year": doc.published_year,
    }

    if dest.exists() and dest.stat().st_size > 0 and not force:
        return {**row, "parse_status": "pending", "status": "cached"}

    try:
        size, sha = _fetch(client, doc.url, dest)
        return {**row, "parse_status": "pending", "file_sha256": sha, "status": "downloaded", "bytes": size}
    except Exception as exc:  # noqa: BLE001
        # A download failure is recorded in the same table as a parse failure.
        # One place to answer "what happened to document X".
        return {
            **row,
            "local_path": None,
            "parse_status": "failed",
            "parse_error": f"download: {type(exc).__name__}: {exc}",
            "status": "failed",
        }


def download_corpus(docs: list[SourceDoc], workers: int = 8, force: bool = False) -> dict[str, int]:
    """Fetch every document in the manifest, concurrently."""
    tally = {"downloaded": 0, "cached": 0, "failed": 0}

    with httpx.Client(headers=BROWSER_HEADERS, timeout=120.0, follow_redirects=True) as client:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(download_one, d, client, force) for d in docs]
            with connect() as conn:
                for i, job in enumerate(futures.as_completed(jobs), start=1):
                    result = job.result()
                    status = result.pop("status")
                    result.pop("bytes", None)
                    tally[status] += 1
                    upsert_document(conn, result)
                    if status == "failed":
                        print(f"  [{i}/{len(jobs)}] FAILED {result['doc_id']}: {result['parse_error'][:110]}")
                    elif i % 25 == 0 or i == len(jobs):
                        print(f"  [{i}/{len(jobs)}] {tally}")
    return tally
