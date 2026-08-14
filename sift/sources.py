"""Corpus discovery: turn four public-sector websites into a list of PDFs.

Every one of these needed probing before it worked, and the quirks are recorded
here rather than in a commit message, because they are the actual content of
"ingesting messy real-world data":

  * GAO (gao.gov) sits behind Akamai and returns 403 to plain curl. Rather than
    fight a bot-detector we take GAO reports from **govinfo.gov**, the GPO's
    official mirror, which has a real API and serves the same PDFs. Its GAOREPORTS
    collection reaches back to 1995 -- and 1990s reports are page scans, which is
    exactly the hard input this project needs.
  * NIST blocks HEAD requests on nvlpubs but allows GET. Any downloader that
    "checks if the file exists" before fetching it will conclude the entire NIST
    catalogue is missing.
  * NIST has no publications API, so we walk the CSRC search UI (100 per page)
    and follow each publication page to its nvlpubs PDF link.
  * NASA NTRS has the cleanest API of the four and the worst PDFs: 1960s reports
    carry a text layer produced by decades-old OCR, so they look machine-readable
    while actually being garbage. See docs/FAILURE_MODES.md.
"""

from __future__ import annotations

import dataclasses
import os
import re
import time
from typing import Iterator

import httpx

# Sec-Fetch-* is what actually gets us past GAO's bot check; a browser
# User-Agent alone is not enough.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}


@dataclasses.dataclass(frozen=True)
class SourceDoc:
    """One candidate PDF, before anything has been downloaded."""

    doc_id: str  # stable, filesystem-safe, unique across the corpus
    source: str  # gao | nist | nasa | federal_register
    title: str
    url: str  # direct PDF URL
    published_year: int | None = None
    expect_scanned: bool = False  # a hint from the source, verified later


def _client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(
        headers=BROWSER_HEADERS,
        timeout=timeout,
        follow_redirects=True,
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:120]


# ---------------------------------------------------------------------------
# GAO, via govinfo
# ---------------------------------------------------------------------------

def gao_reports(limit: int, since: str = "1995-01-01T00:00:00Z") -> Iterator[SourceDoc]:
    """GAO reports from the GPO's govinfo API.

    The collections endpoint needs an API key; DEMO_KEY works but is rate
    limited, so we page at 100 and only hit it ceil(limit/100) times. The PDFs
    themselves come from www.govinfo.gov and need no key at all.
    """
    api_key = os.getenv("GOVINFO_API_KEY", "DEMO_KEY")
    offset, seen = 0, 0
    with _client() as client:
        while seen < limit:
            page_size = min(100, limit - seen)
            resp = client.get(
                f"https://api.govinfo.gov/collections/GAOREPORTS/{since}",
                params={"offset": offset, "pageSize": page_size, "api_key": api_key},
            )
            resp.raise_for_status()
            packages = resp.json().get("packages", [])
            if not packages:
                return
            for pkg in packages:
                pkg_id = pkg["packageId"]
                date = pkg.get("dateIssued") or ""
                year = int(date[:4]) if date[:4].isdigit() else None
                yield SourceDoc(
                    doc_id=_slug(pkg_id),
                    source="gao",
                    title=pkg.get("title") or pkg_id,
                    url=f"https://www.govinfo.gov/content/pkg/{pkg_id}/pdf/{pkg_id}.pdf",
                    published_year=year,
                    # Pre-2000 GAO reports are scans of paper originals.
                    expect_scanned=bool(year and year < 2000),
                )
                seen += 1
                if seen >= limit:
                    return
            offset += len(packages)
            time.sleep(0.5)  # be polite to a DEMO_KEY-rate-limited API


# ---------------------------------------------------------------------------
# NIST, via the CSRC publication search UI
# ---------------------------------------------------------------------------

_PUB_LINK = re.compile(r'href="(/pubs/[^"]+)"')
_NVLPUB_PDF = re.compile(r'href="(https://nvlpubs\.nist\.gov/[^"]+\.pdf)"')
_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
_NIST_DATE = re.compile(r'id="pub-release-date"[^>]*>([^<]{4,40})')


def nist_publications(limit: int, series: str = "SP") -> Iterator[SourceDoc]:
    """Walk CSRC search pages, then resolve each publication page to its PDF.

    Two requests per document is unavoidable -- CSRC's search results do not
    contain the nvlpubs link. We keep it civil with a short sleep.
    """
    seen, page = 0, 1
    search = "https://csrc.nist.gov/publications/search"
    with _client() as client:
        while seen < limit and page <= 30:
            resp = client.get(
                search,
                params={
                    "sortBy-lg": "Date Published DESC",
                    "status-lg": "Final",
                    "series-lg": series,
                    "ipp": 100,
                    "page": page,
                },
            )
            resp.raise_for_status()
            pub_paths = sorted(set(_PUB_LINK.findall(resp.text)))
            if not pub_paths:
                return

            for path in pub_paths:
                if seen >= limit:
                    return
                try:
                    pub = client.get(f"https://csrc.nist.gov{path}")
                    pub.raise_for_status()
                except httpx.HTTPError:
                    continue

                pdfs = _NVLPUB_PDF.findall(pub.text)
                if not pdfs:
                    continue  # withdrawn, or HTML-only -- skip quietly
                pdf_url = pdfs[0]

                title_match = _TITLE.search(pub.text)
                title = (
                    title_match.group(1).split("|")[0].strip()
                    if title_match
                    else path.strip("/")
                )
                # The publication year is in the page, not the PDF URL. The
                # earlier version regexed the URL for a year, found none, and
                # left every NIST document with a NULL year.
                year = None
                date_match = _NIST_DATE.search(pub.text)
                if date_match:
                    year_match = re.search(r"\b(19|20)\d{2}\b", date_match.group(1))
                    if year_match:
                        year = int(year_match.group(0))

                yield SourceDoc(
                    doc_id=_slug("nist-" + path.replace("/pubs/", "").replace("/", "-")),
                    source="nist",
                    title=title,
                    url=pdf_url,
                    published_year=year,
                )
                seen += 1
                time.sleep(0.2)
            page += 1


# ---------------------------------------------------------------------------
# NASA NTRS -- our main supply of genuinely scanned documents
# ---------------------------------------------------------------------------

def nasa_reports(
    limit: int, query: str = "aerodynamics", year_end: int | None = 1975
) -> Iterator[SourceDoc]:
    """NASA technical reports.

    Defaulting year_end to 1975 is intentional: those are microfiche scans, and
    they are the documents that make this corpus honest. Pass year_end=None for
    modern, well-behaved PDFs.
    """
    seen, page = 0, 1
    with _client() as client:
        while seen < limit and page <= 20:
            params: dict[str, object] = {"q": query, "page.size": 100, "page.from": (page - 1) * 100}
            if year_end:
                params["published.lt"] = f"{year_end}-01-01"
            resp = client.get(
                "https://ntrs.nasa.gov/api/citations/search",
                params=params,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return

            for item in results:
                if seen >= limit:
                    return
                downloads = item.get("downloads") or []
                pdf_link = next(
                    (d["links"]["pdf"] for d in downloads if d.get("links", {}).get("pdf")),
                    None,
                )
                if not pdf_link:
                    continue
                ntrs_id = item.get("id")
                # The publication date is nested inside `publications`, not on
                # the record itself. Reading the top-level field returns None
                # for every document, which silently disabled every year filter
                # on the NASA half of the corpus.
                year = None
                for pub in item.get("publications") or []:
                    date = pub.get("publicationDate") or ""
                    if date[:4].isdigit():
                        year = int(date[:4])
                        break
                if year is None:
                    date = item.get("distributionDate") or ""
                    year = int(date[:4]) if date[:4].isdigit() else None
                yield SourceDoc(
                    doc_id=f"nasa-{ntrs_id}",
                    source="nasa",
                    title=(item.get("title") or f"NTRS {ntrs_id}").strip(),
                    url=f"https://ntrs.nasa.gov{pdf_link}",
                    published_year=year,
                    expect_scanned=bool(year and year < 1990),
                )
                seen += 1
            page += 1


# ---------------------------------------------------------------------------
# Federal Register
# ---------------------------------------------------------------------------

def federal_register(limit: int, since: str = "2024-01-01") -> Iterator[SourceDoc]:
    """Rules and notices. Formatting is inconsistent by design -- these are
    typeset from many agencies -- which is good stress on the chunker."""
    seen, page = 0, 1
    with _client() as client:
        while seen < limit and page <= 20:
            resp = client.get(
                "https://www.federalregister.gov/api/v1/documents.json",
                params={
                    "per_page": 100,
                    "page": page,
                    "order": "newest",
                    "conditions[publication_date][gte]": since,
                    "conditions[type][]": "RULE",
                    "fields[]": ["document_number", "title", "pdf_url", "publication_date"],
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return
            for item in results:
                if seen >= limit:
                    return
                if not item.get("pdf_url"):
                    continue
                date = item.get("publication_date") or ""
                yield SourceDoc(
                    doc_id=f"fr-{item['document_number']}",
                    source="federal_register",
                    title=(item.get("title") or item["document_number"]).strip(),
                    url=item["pdf_url"],
                    published_year=int(date[:4]) if date[:4].isdigit() else None,
                )
                seen += 1
            page += 1


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

# Deliberately weighted toward GAO and NIST (the corpus this project claims to
# be about), with NASA carrying the scanned-document load.
DEFAULT_MIX = {
    "gao": 0.40,
    "nist": 0.24,
    "nasa": 0.20,  # ~all scanned
    "federal_register": 0.16,
}

_FETCHERS = {
    "gao": gao_reports,
    "nist": nist_publications,
    "nasa": nasa_reports,
    "federal_register": federal_register,
}


def build_manifest(total: int, mix: dict[str, float] | None = None) -> list[SourceDoc]:
    """Collect `total` candidate documents across all sources.

    A source that is down or has changed its HTML must not take the corpus with
    it, so failures are reported and skipped rather than raised.
    """
    mix = mix or DEFAULT_MIX
    docs: list[SourceDoc] = []
    for name, share in mix.items():
        want = max(1, round(total * share))
        try:
            got = list(_FETCHERS[name](want))
            print(f"  {name:>17}: {len(got):>4} documents discovered (wanted {want})")
            docs.extend(got)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:>17}: DISCOVERY FAILED -- {type(exc).__name__}: {exc}")

    # De-duplicate on doc_id; sources can overlap (govinfo carries some NIST).
    unique: dict[str, SourceDoc] = {}
    for doc in docs:
        unique.setdefault(doc.doc_id, doc)
    return list(unique.values())
