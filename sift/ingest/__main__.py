"""CLI: python -m sift.ingest [--limit N] [--workers N] [--skip-download]"""

from __future__ import annotations

import argparse

from sift.db import healthcheck
from sift.ingest.pipeline import run_ingest


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
