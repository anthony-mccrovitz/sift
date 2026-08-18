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

## 12. A missing binary disguised itself as 24 broken documents

**Symptom.** A full ingest completed successfully: 437 parsed, 42 failed, a
tidy failure log. Every failure in it was real. But 24 of them read
`PDFInfoNotInstalledError: Unable to get page count. Is poppler installed?` —
and those 24 were almost exactly the scanned documents, which are the ones this
project exists to handle.

**Cause.** `pdf2image` shells out to poppler's `pdfinfo`, and poppler was not on
`PATH` for that run. Not because it was uninstalled — because the run was
launched through a login shell, macOS's `path_helper` rebuilt `PATH` from
`/etc/paths`, and `/opt/homebrew/bin` was not in it.

**Handling.** A `preflight()` check that runs before anything is parsed and
refuses to start when `pdfinfo`, `pdftoppm` or `tesseract` is missing, naming
the package to install rather than the binary that was probed.

**Why it matters.** This is the most instructive failure here, because the
pipeline behaved exactly as designed and that was the problem. It is built never
to die on a single document, so a missing binary did not stop it — it marked
every scanned document `failed` and carried on. The result was a green run, a
plausible failure log, and a corpus quietly containing none of the scanned
documents.

The general shape: per-document error handling makes environment errors look
like data errors. One document failing to parse is data. Every document of one
kind failing identically is the machine, and the two belong in different
channels — the failure log describes documents, so it should never be where you
find out poppler is missing. Resilience without a preflight is just a slower way
to get a wrong answer.

---

## 13. The same corpus and the same code produced two different MRRs

**Symptom.** Two evaluation runs against an untouched database returned MRR
0.9115 and 0.9125. Eight of thirty-six questions retrieved a different set or
order of documents; on q012 the gold document moved from rank 6 to rank 5.

**Cause.** Both retrievers ended in `ORDER BY <score> LIMIT k` with no
tiebreaker. `ts_rank_cd` ties constantly — every chunk matching the same terms
the same number of times scores identically, and an identifier lookup gives a
whole document the same `100 + epsilon`. When hundreds of rows tie, *which* `k`
of them survive the `LIMIT` is entirely the executor's choice, and that choice
moves with the query plan.

This is not the same bug as #8. That one was tie-breaking inside rank fusion,
and it was fixed. This one is upstream: the ranked lists *going into* fusion were
already arbitrary, so fusing them deterministically produced a stable function of
unstable inputs.

**Handling.** A total order on both retrievers. The lexical side takes
`ORDER BY score DESC, c.id`, which is free because that plan already sorts. The
dense side cannot: adding `c.id` to the ordering makes a compound sort key the
HNSW index cannot serve, and `EXPLAIN` confirms the planner abandons the index
for a full sequential scan. So the dense query became two-stage — an inner
index-ordered `LIMIT`, an outer `ORDER BY distance, id`.

Distance ties are real here rather than theoretical: government PDFs repeat
boilerplate headers and footers, identical text embeds to an identical vector,
and identical vectors are exactly equidistant from any query.

**Why it matters.** Two things, and the second is the one worth keeping.

First, the number I had published was luck. The full-corpus MRR I recorded as
0.8578 is 0.8812 once retrieval is deterministic — I had been reporting one draw
from a distribution as though it were a measurement.

Second, and worse: my first attempt at a guard against this did not work. I
added a check that ran the evaluation twice and compared. It passed — with the
bug still present. Consecutive runs pick the same plan, so they agree with each
other while both remain arbitrary. Two runs agreeing is not evidence of
determinism; it is evidence that nothing perturbed them.

The check that works runs the second pass under deliberately different planner
settings (`max_parallel_workers_per_gather=4` with the parallel costs zeroed) and
asks whether the answer depends on how Postgres chose to execute it. Reintroduce
the missing tiebreak and it fails ten questions. That is the difference between
a test and a test that can fail.

---

## 14. Loading the same fixture twice does not build the same index

**Symptom.** Found while verifying the fix for #13, by building the database
twice from the byte-identical committed fixture and diffing the results. Three of
thirty-six questions retrieved different documents. Recall and MRR happened to be
unchanged — the differences did not land on a gold document — which is luck, not
a guarantee.

**Cause.** HNSW is an approximate index, and its graph depends on a random level
assigned to each element at insert time. pgvector does not draw that from the
generator `setseed()` controls, so there is no session-level seed that makes the
build reproducible. I tried; it changed nothing, which is how I know rather than
assume.

**Handling.** Partial, and worth being precise about. Raising `hnsw.ef_search`
makes the search walk more of the graph, so it depends less on the graph's exact
shape. Measured across two clean builds:

| `ef_search` | questions whose dense results changed |
|---|---|
| 40 (pgvector default) | 10 of 36 |
| 200 | 3 of 36 |
| 500 | 1 of 36 |

500 is tempting and wrong. On the 14k-chunk fixture it costs nothing; on the full
55k-chunk corpus it takes p50 from 199ms to 262ms, past the 250ms budget the
project set itself. The setting is 200, where the full corpus sits at 225ms.
Recall@6 and MRR are identical at every value tested — this bought stability, not
accuracy.

**Why it matters.** This one is unresolved and stays that way honestly. Two
different things were being called "deterministic":

- *Given this index, does the same query return the same answer?* Now yes, and
  the gate enforces it under a perturbed query plan (#13).
- *Given the same data, do two builds produce the same index?* **No.** Reduced
  from 10 questions to 3, not eliminated, and not eliminable without an exact
  index.

The gate compares metrics to thresholds, not document lists to a golden file, so
it tolerates this. But "our eval is deterministic" would be an overclaim, and the
distinction above is the whole reason to say so carefully. A benchmark's error
bars are part of the benchmark.

---

## 15. The gate failed one build in four, and it was the gate's own fault

**Symptom.** Found by running the full CI sequence — create database, apply
schema, load fixture, evaluate — six times in a row rather than once. One build
in four failed on latency: p50 330ms against a 250ms budget, while the others sat
near 100ms. Recall, MRR and determinism were identical every time.

**Cause.** Not a cold cache, which was the first guess and the wrong one. A cold
cache makes the *first* queries slow; here every query was slow. It was
autovacuum, waking up on a freshly bulk-loaded 14k-row table and competing for
I/O with the queries being timed. CI loads the fixture and starts timing
immediately, so it hits this window every run and loses it sometimes.

**Handling.** The loader now runs `VACUUM ANALYZE` itself after the bulk insert,
which is ordinary practice after any bulk load and takes a couple of seconds.
Doing the housekeeping deliberately beats racing a background worker that does it
whenever it feels like it. Six consecutive clean builds then landed at p50 97–101ms.

I also tried a warm-up pass over the question set before timing, on the cold-cache
theory. It did **not** fix it — one build in five still failed — so it was
removed rather than left in place looking like the fix. A mitigation that does not
mitigate is worse than none, because the next person reads it as a solved problem.

**Why it matters.** This is the same lesson as #13 arriving from a different
direction. A gate that fails a quarter of the time for reasons unrelated to the
pull request does not enforce quality — it teaches the team to re-run CI until it
goes green, which is precisely the habit that lets a real regression through.

The only reason this was found at all is that the sequence was run six times
instead of once. A single green run is not evidence that a gate is stable; it is
one sample. Flakiness is a property you have to go looking for, and the looking is
cheap compared to what it costs to discover it in the middle of a real
regression.

---

## 16. I reimplemented a metric and got the definition wrong

**Symptom.** The test written to prove a reimplementation was faithful failed on
its first run, on three of six cases.

**Cause.** Two of RAGAS's four metrics never needed an LLM —
`IDBasedContextPrecision` and `IDBasedContextRecall` compare retrieved document
ids against reference ids, and the eval set already carries those ids. Moving
them into the free tier meant computing them directly, because importing RAGAS
and the langchain stack would add roughly ninety seconds of install to a gate
whose whole value is being cheap enough to run on every pull request.

My version divided by the number of retrieved *passages*. RAGAS deduplicates to
distinct document ids first. With six chunks drawn from a handful of documents,
a document appears twice most of the time, so the two disagreed constantly —
0.5 against 0.4 on the first case that hit it.

**Handling.** RAGAS's definition ships under RAGAS's name. The passage-level
number was worth keeping, so it stayed under a name that says what it is,
`passage_precision`, with a test asserting the two remain different. They answer
different questions: how many distinct retrieved documents were relevant, versus
how much of the model's context window was spent well.

**Why it matters.** The reimplementation was a reasonable trade and it was also
wrong, and those are not in tension. What made it safe was deciding up front
that a reimplemented metric has to prove agreement rather than assert it, and
writing that test before trusting the number.

Worth noting what the failure was *not*: nothing crashed, and both numbers were
plausible. A 0.4 and a 0.5 both look like reasonable context precision. Had the
test not existed, the metric would have gone into a threshold file, gated
merges, and been quoted in a README under a name that implied a definition it
did not implement. The class of bug this repository keeps finding is the one
where everything looks fine.

---

## 17. Two thirds of one source carried other documents' text

**Symptom.** None. Nothing failed, nothing looked wrong, and the metrics were
fine. It surfaced only because a worksheet built for hand-checking the eval set
sorted questions by how well each gold document supported its own stated answer,
and q017 came last at 0.643. The passage answering it carried the footer
`[FR Doc. 2026-16630]` while being stored under `fr-2026-16687`.

**Cause.** The Federal Register API gives a `pdf_url` per document, but the PDF
behind it is a *page extract* from that day's issue. A rule beginning halfway
down a page inherits the top of that page; one ending halfway down carries
whatever starts underneath. So downloading document X also gets you the tail of
X-1 and the head of X+1.

Measured across the corpus: **54 of 80** Federal Register documents held at least
one chunk belonging to another document. `fr-2026-16687` was 37 chunks, of which
chunks 0-2 were the end of `fr-2026-16630` and chunks 34-36 were the beginning of
an unrelated FAA airworthiness directive. Only 3-33 were the document.

**Handling.** Every Federal Register document ends with its own filing footer —
`[FR Doc. 2026-16687 Filed 8-13-26; 8:45 am]` — which is unambiguous and
machine-readable. That marks the end; the last *foreign* footer before it marks
where the previous document stopped. The trim happens on the element stream
before chunking rather than after, because chunk boundaries computed across a
neighbour's text bake foreign sentences into our chunks, and no later filter gets
them back out.

54 affected documents dropped to 6, and 313 chunks left the corpus. The remaining
6 are the ones whose own footer never appears in the extracted text, where
trimming would be guessing; `document_span` keeps them whole and returns
`own-footer-not-found` so the reason travels with the decision instead of being
absent from a log.

**Why it matters.** Every chunk carries its `doc_id` into retrieval and back out
as a citation. Foreign text under the wrong `doc_id` means the system can quote a
passage, attribute it to a document and a page, and be pointing at a different
document entirely — a confident, sourced, wrong answer, which is the one failure
this project exists to prevent. It had been doing that on 68% of one source for
the entire life of the repository, through every green CI run.

And the part worth keeping. This README explained q017's miss as a limitation of
dense retrieval without query rewriting. After correcting the gold document to
`fr-2026-16630`, q017 *still* misses — retrieval really does return GAO reports
for "General Accounting Office". The stated explanation was right. The evidence
offered for it was a mislabelled chunk, and the two facts were independent.

Being accidentally right is not the same as knowing. A plausible explanation that
happens to be correct is indistinguishable, from the outside, from one that is
not — and the only thing that separates them is having checked. That is the
entire argument for the hand-verification pass, made by the tool built to do it,
before a single question had been verified by hand.

---

## 18. Trying to avoid the API key, and failing on purpose

**Symptom.** Not a bug -- an experiment that did not work, recorded because the
result is the argument for the decision it settled.

Three of the answer-level metrics turned out not to need a paid judge at all,
and moving them to a local generator worked (see the local provider in
`sift/llm.py`). So the obvious next question was whether the same trick would
retire the last two: run RAGAS itself against a local model and stop needing a
key at all.

**What happened.** The pipeline ran -- dataset built, answers generated, metrics
instantiated, all eight jobs dispatched for two questions. Then every one of
them failed:

    Exception raised in Job[0]: TimeoutError()
    ... Job[1] through Job[7], identically

RAGAS times each job out at 180 seconds, which is generous for a hosted judge
and nowhere near enough for a 3B model on an M1 Pro. Raising the timeout to 1800
seconds and cutting the run to a *single* question still had not finished after
ten minutes.

**Why it settles the question.** The full run is 32 answerable questions across
four metrics, and faithfulness alone issues roughly one call per claim in each
answer. At the observed rate that is hours, not minutes, per run -- for a gate
that is supposed to run on a pull request.

So the objection to a local judge is no longer only the one worth making on
principle (a small model grading its own output produces numbers that look
authoritative and mean nothing, which is the exact failure this repository
documents seventeen times). It is also simply not fast enough to finish. Two
independent reasons, one of them now measured rather than asserted.

The experimental branch was reverted rather than left in place. Code that cannot
do the job it was added for is worse than no code, because the next person reads
its presence as evidence the path works -- the same reasoning that removed the
warm-up pass in #15.

**What it leaves.** `faithfulness` and `answer_relevancy` require an API key,
and that is now a measured constraint rather than an assumption. Everything else
in this project runs without one.

---

## Still open

- **q017 retrieval miss.** "What change did the rule make to references to the
  General Accounting Office?" retrieves GAO reports, because the phrase "General
  Accounting Office" is overwhelmingly associated with GAO documents. The answer
  is in a Federal Register rule that renames it. Left failing on purpose — it is
  a genuine limitation of lexical + dense retrieval without query rewriting, and
  it stayed a miss after its gold document was corrected (see #17), which is what
  makes that claim checked rather than merely plausible.
- **6 Federal Register documents still carry a neighbour's text.** Their own
  filing footer never appears in the extracted text, so the boundary cannot be
  established without guessing. Down from 54; see #17.
- **The eval set has still not had its human pass.** `scripts/verify_eval_set.py`
  generates the worksheet, and finding #17 came out of building it, but only
  q017 has actually been corrected. The other 35 remain as drafted.
- **Rotated sidebar text still enters chunks.** Federal Register pages produce
  leading fragments like `1 S E L U R h t i` (a vertical "RULES" banner). It is
  filtered from the *probe* but not from chunk text. Low impact, not yet fixed.
- **The RAGAS judge shares a model with the generator.** A model grading its own
  output is measurably more generous. Faithfulness is least affected, which is
  why it carries the strictest threshold, but these are regression detectors,
  not absolute scores.
