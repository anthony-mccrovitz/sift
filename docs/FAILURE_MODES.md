# What broke

Everything below happened while building this, in this order. Each entry is the
symptom first, because that is how you actually meet these problems.

---

## 1. GAO returns 403 to anything that is not a browser

**Symptom.** `curl https://www.gao.gov/products/gao-26-109046` returns
`403 Access Denied` from Akamai. A browser User-Agent alone does not fix it.
Oddly, `https://www.gao.gov/rss/reports.xml` works fine, which sends you looking
for a problem with the specific URL rather than with bot detection.

**Cause.** Akamai scores the whole request, not just the UA. The missing signal
was the `Sec-Fetch-*` header family that real browsers always send.

**Handling.** Two changes. Adding `Sec-Fetch-Dest/Mode/Site` gets a 200. But for
the corpus we do not scrape gao.gov at all -- we take GAO reports from
**govinfo.gov**, the GPO's official mirror, which has a documented API, no bot
wall, and the same PDFs. Fighting a bot detector for 200 documents is a bad
trade when an official API exists.

**Bonus.** govinfo's GAOREPORTS collection goes back to 1995, and 1990s reports
are scans of paper. That turned a workaround into the source of the hardest and
most useful documents in the corpus.

---

## 2. NIST serves PDFs but blocks HEAD

**Symptom.** `curl -I https://nvlpubs.nist.gov/.../NIST.SP.800-53r5.pdf` returns
`404`. The same URL with `GET` returns a 6MB PDF.

**Cause.** nvlpubs rejects HEAD requests.

**Handling.** Never probe with HEAD. This is worth stating because the obvious
downloader design -- "check the file exists, then fetch it" -- concludes that
the entire NIST catalogue is missing.

---

## 3. HTTP 200 with an HTML "Page Not Found" body

**Symptom.** Three GAO downloads produced files that were not PDFs. Status 200,
`content-type: text/html`.

**Cause.** Some 1990s GAO packages have no PDF rendition at all -- only text and
zip. govinfo serves its normal 200 HTML "Page Not Found" page for the PDF path.

**Handling.** Validate the `%PDF` magic number rather than the status code, and
classify the body: "no PDF rendition published (soft 404)" is permanent, while a
throttling page is worth retrying. Related bug found at the same time: the
retry decorator was retrying `NotAPdf` three times with exponential backoff,
spending ~20 seconds per document on a condition that can never change. Only
transport errors are retried now.

**Cost.** ~1.2% of the corpus. Recorded in `data/failed_documents.log` rather
than dropped, because "487 of 500" is only meaningful if you can say which 13.

---

## 4. The scanned documents that look machine-readable

This is the interesting one.

**Symptom.** 1960s NASA technical reports have a text layer, so
`strategy="auto"` picks the fast pdfminer path and returns text with no error.
The text is unusable:

```
N67-30580 UNSTEADYAERODYNAMICS By M. F. Platzer SUMMARY This study
briefly describes some of the problem areas gram in the Saturn launch
vehicle development that are related to unsteady phenomena. pro-
```

Words are glued together, and `pro-` / `gram` are separated by 20 words because
a hyphenated word broke across columns. Nothing raises an exception. Retrieval
just quietly gets worse.

**Cause.** These are microfiche scans that were OCR'd decades ago and shipped
with that OCR embedded. The text layer exists and is garbage.

**Detection, after two failed attempts.** The signal is *layout*, not spelling:

- *First attempt* — count glued words and lone letters. Failed: the NASA scan
  scored 0.14 ("clean") because its individual words are fine.
- *Second attempt* — median words per line. Correctly flagged NASA, but
  produced a **false positive** on NIST SP 800-53. Cause: NIST publications
  carry a rotated sidebar watermark that pdfminer extracts one character per
  line, ~130 junk lines per page, dragging the median to 1.
- *What works* — drop lines under 3 characters (kills the rotated-text
  artifact), then measure what fraction of characters live in lines of 4+ words:

  | document | prose fraction |
  |---|---|
  | GAO 1995 report | 0.95 |
  | NIST SP 800-53r5 | 0.93 |
  | NASA 1967 scan | **0.17** |

**Also required:** sample pages from a third of the way in, never from the
front. Title pages and tables of contents are legitimately one word per line, so
a document probed at page 1 looks exactly like a bad scan.

---

## 5. hi_res is worse than ocr_only on real scans

**Symptom.** After correctly routing scans to OCR, the output was still
scrambled -- but differently. Correct words, well-formed lines, wrong order:

```
II. PANELFLUTTER INTRODUCTION I. are that are aerodynamics airflows
Unsteady that may re- and dependent produce time a dynamic control sponse
```

**Measurement.** Rather than guess, `scripts/compare_strategies.py` runs all
three strategies on the same document and scores readability by counting common
English bigrams ("of the", "in the") per 1000 words -- a metric that survives
correct-words-wrong-order, which layout metrics do not:

| strategy | seconds | garbled score | bigrams/1k |
|---|---|---|---|
| fast | 11.7 | 0.81 | 48.5 |
| hi_res | 40.1 | 0.05 | **23.9** |
| ocr_only | 36.3 | 0.04 | **50.4** |

**Cause.** `hi_res` runs layout detection before OCR. On a low-contrast
microfiche scan it carves the page into regions and emits them in the wrong
reading order. It produces the *cleanest-looking* output and the least readable
text — the failure mode that looks like success. `ocr_only` skips region
detection and reads straight through.

**Handling.** Scanned documents route to `ocr_only`. It is faster *and* better
here. Full-corpus ingest time dropped from 6.0 to 1.9 minutes on the 25-document
smoke corpus as a side effect.

**Tradeoff accepted:** `ocr_only` does not reconstruct table structure. Right
for this corpus (1960s technical prose); wrong for a corpus of scanned financial
tables. `force_strategy` exists for that case.

---

## 6. Lexical search silently returned nothing

**Symptom.** `keyword_search("what weaknesses did GAO find in export controls
for missile technology")` returned **zero rows**. Hybrid retrieval still
returned results, so nothing looked broken — it had quietly degraded to
vector-only.

**Cause.** `websearch_to_tsquery` and `plainto_tsquery` both join terms with
AND. The query demanded that a single 1800-character chunk contain *every* word
including "what", "did" and "find".

**Handling.** Strip question scaffolding, then OR the content terms. Recall
comes from the OR; precision comes from `ts_rank_cd`, which rewards chunks
matching more terms, closer together. Quoted phrases are preserved.

**Why it matters more than the fix.** This is the second silent degradation in
this list. Both were found by evaluation, not by tests or exceptions — which is
the argument for the CI gate in one sentence.

---

## 7. Document titles were not searchable

**Symptom.** Eval showed `identifier` category recall of **0.33**. "What is NIST
SP 1271?" retrieved three other NIST documents but not SP 1271.

**Cause.** The full-text index covered `chunks.text` only. "SP 1271" appears in
the document's *title* and essentially nowhere in its body.

**Handling.** Denormalise `doc_title` onto `chunks` and build the generated
`tsvector` from `setweight(title,'A') || setweight(text,'B')`, so a title match
outranks an incidental body mention. A generated column cannot reach into
another table, which is why the title is copied rather than joined.

**Also:** bare identifiers like `GAO/NSIAD-95-82` matched nothing, because the
corpus stores `gaoreports-nsiad-95-82`. Identifiers are now split on their
separators and rejoined with wildcards (`%nsiad%95%82%`) to match `doc_id`
directly.

**Result:** identifier recall 0.33 → 1.00; overall recall@6 0.906 → 0.969.

---

## 8. Rank fusion depended on row insertion order

**Symptom.** MRR moved between 0.953 and 0.932 on an *identical* corpus, purely
from reloading the database.

**Cause.** Exact RRF score ties are common — two chunks each ranked 1 by one
retriever score identically. Python's `sorted` is stable, so the order the runs
happened to arrive in decided the ranking, and `TRUNCATE`+reload renumbered the
rows.

**Handling.** Tie-break on `chunk_id`: `sorted(key=lambda h: (-h.fused_score,
h.chunk_id))`. Found by a unit test asserting fusion is order-independent, not
by observation.

**Why it matters.** A regression gate on a nondeterministic metric is noise. The
honest MRR after the fix is 0.9167 — lower than the number I would have
published before, and the first one that means anything.

---

## 9. Cold model load poisoned the latency gate

**Symptom.** `retrieval_p95_ms` of 2793ms against a budget of 800ms, while p50
was 49ms.

**Cause.** `sentence-transformers` loads lazily. The first query absorbed ~2.5s
of model load, which landed in the p95 of a 36-question run.

**Handling.** Warm the model before timing anything, in both the eval harness
and the FastAPI lifespan. p95 went to 67ms. The load time is real, but it is a
startup cost, not a query cost, and conflating them makes the gate a coin flip
on whether CI's disk cache was warm.

---

## 10. The quality gate would have skipped green instead of failing red

**Symptom.** None, locally. `make eval` passed, the unit tests passed, and CI
was configured. `import ragas` raised `ModuleNotFoundError: No module named
'langchain_community.chat_models.vertexai'`.

**Cause.** `ragas==0.4.3` does `from langchain_community.chat_models.vertexai
import ChatVertexAI` at module import time. `langchain-community` 0.4 deleted
that module. Only `langchain-anthropic`, `langchain-openai` and
`langchain-huggingface` were pinned, so pip resolved the unpinned
`langchain-community` to 0.4.2 and the import died. Pinning it back to 0.3
forces `langchain-core` <1.0, which drags the entire langchain family to the
0.3 line — the pins are load-bearing as a set, not individually.

**Handling.** Pin `langchain-community==0.3.31` and `langchain-core==0.3.86`
alongside the provider packages, in `requirements.txt` *and* in the workflow's
inline install list. Then add an explicit `python -c "import ragas"` step to CI.

**Why it matters.** This is the worst failure in the project, and it is not a
retrieval bug. The RAGAS job was guarded by "is an API key present?" — so a
gate that could not even import would have reported **skipped**, which renders
green. A quality gate that silently does not run is strictly worse than no
gate, because it produces the belief that quality is being checked. The fix is
one line of pinning; the lesson is that a gate must be able to fail loudly, and
"skipped" must never be reachable through a broken dependency.

---

## 11. A six-hour ingest could not survive a laptop going to sleep

**Symptom.** The full ingest died twice at 220/481 and 103/261 documents. Both
times, restarting meant reparsing everything from zero.

**Cause.** Two separate things. The first death was the Postgres container
stopping when the machine slept overnight. The second was the parent process
being terminated while the work itself was fine. The real problem was neither:
`run_ingest` rebuilt its work list from the manifest every time, so it had no
notion that 220 documents were already parsed and sitting in Postgres.

**Handling.** Added `--resume`, which skips documents whose `parse_status` is
`parsed` or `empty`. `empty` counts as done because rerunning an identical
parse will not produce text that was not there the first time; genuine failures
are still retried, since those include transient problems. Kept it opt-in —
after changing chunking you want everything reprocessed, and a resume that
silently kept stale chunks would be the worse bug. Also detached the long run
from the controlling terminal and held idle sleep off for its lifetime.

**Why it matters.** It forced a second fix that was not obvious. The failure log
was written from the current run's results, so a resumed run would have
described only the documents it happened to touch — while the README points at
that file as a description of *the corpus*. The log and the closing summary now
read back from the database. Any incremental pipeline has this bug shape: the
moment work becomes resumable, every report derived from "what this run did"
quietly starts lying.

---

## Still open

- **q017 retrieval miss.** "What change did the rule make to references to the
  General Accounting Office?" retrieves GAO reports, because the phrase "General
  Accounting Office" is overwhelmingly associated with GAO documents. The answer
  is in a Federal Register rule that renames it. Left failing on purpose — it is
  a genuine limitation of lexical + dense retrieval without query rewriting.
- **Rotated sidebar text still enters chunks.** Federal Register pages produce
  leading fragments like `1 S E L U R h t i` (a vertical "RULES" banner). It is
  filtered from the *probe* but not from chunk text. Low impact, not yet fixed.
- **The RAGAS judge shares a model with the generator.** A model grading its own
  output is measurably more generous. Faithfulness is least affected, which is
  why it carries the strictest threshold, but these are regression detectors,
  not absolute scores.
