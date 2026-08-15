# Benchmark history

Appended by `scripts/record_benchmark.py` after each evaluation run. The point
is to catch slow drift -- a gate only fails on a single bad pull request, while
a table shows quality bleeding away over ten good-looking ones.

| date | commit | docs | chunks | recall@6 | MRR | p50 ms | faithfulness | note |
|---|---|---|---|---|---|---|---|---|
| 2026-08-15 | `93db16a` | 460/500 | 55294 | 0.9375 | 0.8578 | 194 | - | full 460-document corpus, 29 scanned, after OCR retry |
| 2026-08-15 | `9b3bfb4` | 460/500 | 55294 | 0.9375 | 0.8812 | 199 | - | deterministic ORDER BY tiebreaks; embeddings committed to the fixture |
| 2026-08-15 | `9b3bfb4` | 460/500 | 55294 | 0.9375 | 0.8812 | 218 | - | hnsw.ef_search 40 -> 200 to reduce cross-build variation |
