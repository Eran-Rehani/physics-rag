# physics-rag — Complete Guide

A working guide to what this project is, why it exists, how it is built, and where it
falls short. The [README](README.md) is the short version; this is the long one.

Everything stated here as a number was measured on the real corpus and the real hardware.
Where something has not been measured, this document says so rather than guessing.

---

## 1. What it is

A local, citation-grounded RAG system that answers physics questions from a personal
corpus of LaTeX course notes and PDF textbooks — on a CPU-only laptop, with no data
leaving the machine.

Two properties define it, and everything else is in service of them:

1. **Every claim carries a citation** of the form `[filename, section/page]`, resolvable
   back to a real file on disk.
2. **It abstains.** When retrieval confidence is low, it says *"not found in corpus"*
   rather than producing a plausible answer from the model's parametric memory.

```console
$ rag ask "What is the Friedmann equation in cosmology?"

The Friedmann equation binds the curvature of space ($\kappa$), the expansion rate
($a(t)$), and the energy density ($\epsilon$) of the universe
[Introduction to Cosmology-Barbara Ryden-2022.pdf, 4 Cosmic Dynamics > 4.2 The Friedmann Equation].

confidence 0.886
```

### The goal

The corpus is real coursework: graduate and undergraduate physics — E&M, quantum
mechanics, mathematical methods, astrophysics — accumulated across several years as a mix
of `.tex` lecture notes, drill books, past exams, and PDF textbooks. The question that
motivated the project is the mundane one: *where did we prove that, and what was the
notation?*

A general-purpose assistant cannot answer that, because it has never seen these files. A
naive RAG demo can retrieve from them but will happily invent a citation. The point of
this project is the part that is usually skipped — making the citations true, and making
the system decline when it cannot support an answer.

It is also a public portfolio project, so code quality, tests, and documentation are
treated as part of the deliverable rather than as overhead.

### Constraints that shaped everything

| Constraint | Consequence |
|---|---|
| CPU-only i7-12700H, 16 GB RAM | 4.8 GB generation model + 118M embedder, and they **cannot run at the same time** |
| Corpus is private | CI can never see it; tests must run without it |
| Equations must not be corrupted | Chunking is math-aware and provably never splits inside math |
| Corpus is majority Hebrew | An English-only embedder is unusable |

---

## 2. Why generic RAG does not work here

Every one of these was discovered by running against the actual files, not anticipated.

### The notes are majority Hebrew, with English and LaTeX math inline

A section heading looks like `\section{שבוע 2 - הרצאה 3}` with the body in Hebrew prose
and the mathematics in standard LaTeX. The default recommendation for a small local
embedder — `bge-small-en` — is useless on this corpus.

`multilingual-e5-small` (118M params, 384-dim) is used instead. The measured cosine
similarity between an **English query** and its corresponding **Hebrew passage** is
**0.814**, which is what makes "ask in English, retrieve from Hebrew notes, answer in
English" viable at all.

### Hebrew PDF text arrives wrapped in bidi control characters

`pdftotext` returns Hebrew in *logical* order, but wrapped in Unicode bidirectional
controls (U+202A–U+202C). Left in place, the same sentence reached via a `.tex` file and
via its compiled PDF would hash differently, defeating deduplication and content-hash
identity. A shared normalizer strips U+202A–U+202E and U+2066–U+2069 and applies NFC
normalization, and **both parsers use it**, so identical content compares equal regardless
of the route it took.

### PDF extraction destroys mathematics

This is the single most important asymmetry in the project:

| Source | `E² = p²c² + m²c⁴` extracts as | Fidelity |
|---|---|---|
| `.tex` | `S=k_{B}\ln\left(\Omega\right)` — verbatim | **exact** |
| `.pdf` | `E 2 = p2 c2` — superscripts flattened to inline digits | **degraded** |

A system that ignores this will confidently present `E 2 = p2 c2` as an equation. So every
chunk carries `math_fidelity: exact | degraded`, and the generator is instructed never to
reproduce LaTeX from a degraded chunk — it may cite such a chunk for prose, but not for
notation.

This is a mitigation, not a solution. The limitation section is honest about how well a
small model complies.

### Many PDFs are compiled outputs of a `.tex` sitting beside them

`Sol_05.tex` and `Sol_05.pdf` in the same directory are the same content at two different
fidelities. Ingesting both floods retrieval with near-duplicates and inflates any hit-rate
metric. `dedupe_sources` deduplicates on `(directory, stem)` and **prefers the `.tex`
source**, logging every skip.

### The corpus lives on a FUSE mount where `find` times out

`find` over the Google Drive mount **timed out at 120 seconds**. Reading directly off the
mount during ingest is not viable. The corpus is staged locally with `rsync` first, and
ingest is incremental and resumable so an interrupted run costs only the file in flight.

---

## 3. End-to-end walkthrough

### Architecture

```
                    ┌──────────────────────────────────────────┐
  .tex ──► TexParser│  math_fidelity = exact                   │
                    │                                          │
  .pdf ──► PdfParser│  math_fidelity = degraded, page numbers  │
                    └────────────────┬─────────────────────────┘
                                     │  Section(title, level, body, page, path)
                                     ▼
                            normalize  (NFC, strip bidi, collapse blanks)
                                     ▼
                            chunking   section = primary boundary
                                       never splits inside math
                                     ▼
                     Embedder ──► E5Embedder  ("query: " / "passage: " enforced inside)
                                     ▼
                    VectorStore ──► ChromaStore  (hnsw:space = cosine)

  question ──► Retriever ──► confidence ──┬── below threshold ──► "not found in corpus"
                                          │                        (generator never called)
                                          └── above ──► AnswerService ──► Generator
                                                            │              (llama-server HTTP)
                                                            ▼
                                                   answer + [file, section/page]
```

### Tracing one question through `rag ask`

**1. Embed the query.** `E5Embedder.embed_query` prefixes the text with `"query: "` before
encoding. This is not decoration — e5 is an *asymmetric* model trained with distinct
prefixes for queries and passages. Omitting them degrades retrieval with no error and no
warning, so the prefix is applied inside the embedder and callers cannot forget it.

**2. Search Chroma.** `ChromaStore.query` runs a cosine k-NN over the collection. The
collection was created with `metadata={"hnsw:space": "cosine"}`, because **Chroma's default
is L2** — a silent correctness trap, since L2 over unnormalized embeddings ranks differently
and nothing errors.

**3. Deduplicate and score.** `dedupe_results` drops byte-identical text. `Retriever`
returns the ranked chunks plus a confidence score.

**4. The confidence gate.** `AnswerService` — not the retriever — compares confidence
against `abstain_threshold`. Below it, the service returns "not found in corpus" and
**the generator is never called**. This is why out-of-corpus questions return in seconds
while answerable ones take minutes: abstention short-circuits the expensive step.

**5. Assemble the prompt.** `build_context` renders the retrieved chunks into numbered
`SOURCE n -- cite as [label]` blocks. The awkward phrasing is deliberate; see §5.

**6. Generate.** `LlamaServerGenerator` POSTs to llama.cpp's OpenAI-compatible endpoint on
`:8080` at temperature 0.2 — overridden per request, because the server is configured at
1.0 for general use and citation-following work wants near-greedy decoding.

**7. Bind citations.** `resolve_numeric_citations` rewrites any bare `[3]` or `[SOURCE 3]`
the model emitted back into the real label, and `extract_cited_labels` pulls out what was
actually cited so the eval harness can score it.

### Tracing one file through `rag ingest`

```
discover_files      walk root; honour exclude_dirs; skip dotdirs; keep .tex/.pdf
      ▼
dedupe_sources      collapse (directory, stem); prefer the .tex over its compiled .pdf
      ▼
IngestState         skip if (path, size, mtime) unchanged — unless --force
      ▼
parse_file          TexParser or PdfParser → ParsedDocument(sections[])
      ▼
normalize           NFC, strip bidi, flatten newlines inside titles
      ▼
chunk_document      section = primary boundary; sub-split oversized sections math-safely
      ▼
embed_documents     batches of 64, "passage: " prefix applied internally
      ▼
ChromaStore.add     upsert by content-derived chunk_id (idempotent)
      ▼
IngestState.save    atomic temp-file + os.replace, every 25 files
```

The two properties that matter operationally: **idempotent** (re-ingesting the same file
overwrites rather than duplicates, because the ID is derived from content) and
**resumable** (state is checkpointed every 25 files, so a crash costs at most 25 files of
work plus the one in flight).

---

## 4. Module-by-module

| Module | Responsibility | The non-obvious decision it encodes |
|---|---|---|
| `parsers/base.py` | `Section`, `ParsedDocument`, and the `Parser` Protocol | Parsers return **structure**, not text. A `Section` knows its title, level, page and ancestry, so a citation can name a section rather than a byte offset |
| `parsers/tex.py` | `.tex` → sections | Strips the preamble and comments *before* splitting, so `\section` inside a commented-out block or a preamble macro doesn't create a phantom section. Handles starred forms (`\section*`) |
| `parsers/pdf.py` | `.pdf` → sections + page numbers | Shells out to `pdftotext -layout` and splits pages on `\f`, then maps text spans onto the PDF's bookmark outline via `pypdf`. No bookmarks → page-level citations only |
| `normalize.py` | Bidi stripping, NFC, title flattening, content hashing | **Shared by both parsers.** This is what makes `.tex` and PDF renderings of the same content hash equal |
| `chunking.py` | Semantic-section chunking with math-safe sub-splitting | `find_math_spans` computes the intervals a boundary may never land inside; `_choose_split` only ever picks a candidate outside them |
| `embeddings.py` | `Embedder` Protocol; `E5Embedder` | The e5 prefixes are applied **inside** the class and unit-tested. A caller cannot silently degrade retrieval by forgetting them |
| `store.py` | `VectorStore` Protocol; `ChromaStore` | Cosine space set explicitly at creation; `None` metadata dropped (Chroma rejects it); duplicate IDs collapsed within a batch |
| `retrieval.py` | Ranking, confidence, citation formatting | Dedup removes only **byte-identical** text — deliberately conservative, and named as a limitation rather than hidden |
| `generation.py` | `Generator` Protocol; `LlamaServerGenerator` | HTTP, not in-process. The 4.8 GB model lives in a separate process that can be started and stopped independently — which is what makes the 16 GB serialisation workable at all |
| `answer.py` | Abstain-or-cite policy, prompt assembly, citation binding | The **service** owns the abstention decision, not the retriever's config |
| `evaluation.py` | Hit-rate, citation accuracy, abstention F1, threshold sweep | `sweep_threshold` asks each question **once** and recomputes decisions per threshold — a 20-point sweep costs one run, not twenty |
| `ingest.py` | Discovery, dedup, incremental state, orchestration | State is written atomically (temp file + `os.replace`) so an interrupted ingest cannot corrupt it |
| `cli.py` | `rag ingest / ask / stats / eval / ui` | Heavy imports (`torch`, `chromadb`) are deferred inside command bodies so `rag --help` stays instant |

---

## 5. Design decisions and why

### Protocol seams

`Parser`, `Embedder`, `VectorStore`, and `Generator` are `typing.Protocol`s. This buys two
things:

- **Swappability.** Replacing Chroma with another store, or llama.cpp with an API model,
  means writing one class — nothing else changes.
- **Testability without the world.** CI has neither the private corpus nor the 4.8 GB
  model. Deterministic fakes (`tests/fakes.py`) satisfy the same Protocols, so the whole
  pipeline — including abstention logic and citation binding — is exercised on every push.

This is the difference between "swappable in principle" and swappable in fact: the fakes
prove the seams are real, because if a module reached around a Protocol into a concrete
type, CI would fail.

### Semantic-section chunking instead of fixed windows

A fixed 512-token window has no idea where a derivation begins. Section boundaries are
already the author's own semantic partition, so they are the primary split. A section that
exceeds the window is sub-split at paragraph or display-math boundaries, and the parts
share one `section_path` plus `part: i/N` — so the citation still names one section rather
than "chunk 47".

### Math-safe splitting

The spec forbids corrupting equations, so this is enforced rather than hoped for. Chunk
boundaries provably never land inside `$…$`, `\(…\)`, `\[…\]`, or a math environment
(`equation`, `align`, `align*`). Verified on the real corpus:

> **2260 sections, 130 sub-split, zero boundaries inside math, zero text lost.**

### `math_fidelity: exact | degraded`

Covered in §2. The metadata exists so the constraint is *enforceable* — the prompt can
forbid presenting degraded math as exact LaTeX, and the eval can measure whether the model
complied — rather than aspirational.

### Content-hash chunk IDs

`chunk_id = f"{content_hash(text)}-{part}"`. Consequences:

- Re-ingesting an unchanged file is a no-op upsert, so `--force` is safe and resume never
  duplicates.
- Byte-identical content in two files maps to **one** stored chunk, which is desirable —
  a boilerplate section repeated across ten problem sets should not occupy ten slots.

The second property caused a real crash; see §8.

### One collection per course

`physics` (astrophysics) and `qm2` (quantum theory) are separate Chroma collections. The
reason is the abstention threshold: it is calibrated from the *separation* between
answerable and unanswerable confidence bands, and that band shifts with corpus composition.
Mixing corpora blurs both bands into a single uncalibratable middle.

### `SOURCE n -- cite as [label]`

An earlier prompt numbered the context blocks plainly, and the model cited `[3]` — the
block number — instead of the filename. Numbered markers compete with citation syntax. The
current format names the expected output inline, and a rewrite pass resolves any surviving
numbers back to labels.

---

## 6. Evaluation

### What the harness measures

| Metric | Question it answers |
|---|---|
| **Retrieval hit-rate** | Did any retrieved chunk come from an expected file? |
| **Citation accuracy** | Do the emitted `[file, section/page]` labels match the expected source? |
| **Abstention precision / recall / F1** | Measured against deliberately unanswerable questions |
| **`--calibrate`** | Sweeps the abstain threshold and reports the value maximising abstention F1 |

`match_citation` splits an emitted label on the **first** comma — section paths contain
`" > "` and commas of their own — then does a casefolded substring test against the
expected section. Substring rather than equality, so a citation stays correct if the
model reproduces a section path with slightly different leading context.

### Why the threshold is a measurement, not a constant

This is the most transferable result in the project.

`multilingual-e5` cosine scores **compress into a narrow high band, measured at ~0.78–0.91**.
An utterly unrelated question — *"optimal cache eviction policy for a distributed key-value
store"* — still scored **0.801** against a corpus of astrophysics textbooks.

So a plausible-looking default of `0.35` is not conservative. It is **inert**: it never
fires, every question reaches the generator, and the abstention feature silently does not
exist. The current default of `0.82` was derived by sweeping against deliberate negatives.

It is also **corpus-specific**. The 0.82 applies to the astrophysics collection. `qm2` needs
its own sweep, and using the astrophysics number there would be a guess wearing the costume
of a measurement.

### The one completed run — a demonstration, not a score

Six questions (4 answerable, 2 deliberately out-of-corpus) against a 4216-chunk astrophysics
collection:

| Metric | Result |
|---|---|
| Retrieval hit-rate | **1.000** (4/4) |
| Citation accuracy | 0.500 (2/4) |
| Abstention precision / recall / F1 | **1.000** |
| Mean confidence, answerable | 0.889 |
| Mean confidence, negative | 0.794 |
| Calibrated `abstain_threshold` | **0.82** |

The meaningful figure is the **separation** (0.889 vs 0.794), not the averages. Citation
accuracy of 0.5 is a real miss, not a rounding artifact: on two questions the model cited a
source outside the expected set.

### Current status, stated plainly

`eval/eval_set.yaml` holds 20 answerable questions plus 5 negatives. Two caveats matter more
than any number derived from them:

1. **The questions are TOC-derived, not owner-authored.** They were written from the PDFs'
   outline section paths rather than from coursework. Each question was reverse-engineered
   from its own answer key, so retrieval hit-rate on this set is **partly circular** — it
   measures that the pipeline retrieves what it was pointed at, not that it answers a
   question a student would actually ask. It is a smoke test of the harness, not a quality
   score.
2. **The 25-item run has not been completed.** A `--calibrate` run was paused partway. The
   harness assembles its report only after scoring every item, so no partial numbers exist.
   No measured coursework table appears in this project because one does not exist yet, and
   inventing one would defeat the point.

A pre-flight against the store's metadata confirmed all 25 expected `(file, section)` pairs
resolve — **0 unmatched**, en-dashes included — so any citation failure the run eventually
reports is attributable to retrieval or generation rather than a mistyped answer key.

---

## 7. Operational constraints

### Generation and embedding cannot coexist on 16 GB

`llama-server` holds ~4.8 GB plus a 24k-context q8 KV cache. Starting the e5 embedder
alongside it drove available RAM low enough that **the kernel OOM-killed an ingest
mid-run** — no traceback, no summary, just a dead process. Serialise them:

```bash
# ingest (embedder only — llama-server must be stopped)
rag ingest ~/.cache/physics-rag/corpus

# then ask/eval (llama-server up)
llm-serve rag-quality -d      # wait for {"status":"ok"} on :8080
rag eval --calibrate
```

### The system is prompt-bound, not generation-bound

Prompt processing measures **~40 tok/s at 2.2k-token contexts** on this hardware. The cost
of an answer is dominated by the retrieved context, not by the tokens generated. The
practical implication: **keep `top_k` small**. Doubling `top_k` roughly doubles latency
while adding mostly redundant chunks.

### Ingest throughput

**~116 ms/chunk.** Eleven textbook-heavy PDFs produced 4216 chunks in 490 s. Textbook PDFs
are far slower per file than lecture `.tex` files, so throughput varies by an order of
magnitude depending on what a course directory actually contains.

| Collection | Content | Chunks |
|---|---|---|
| `physics` | Year 4 astrophysics, 11 files, 0 failures | 4097 |
| `qm2` | Course 77605 Quantum Theory (2) | in progress |

---

## 8. What testing on fakes cannot catch

Every bug below survived a green test suite and was found only by running the real thing.
This list is the honest counterweight to "94 tests passing".

| # | Bug | Why fakes missed it |
|---|---|---|
| 1 | `pypdf`'s `Destination.children` is a **method**, not a property. Walking it as an attribute raised `TypeError` on every bookmarked PDF — i.e. every textbook | Third-party API shape. No fake imitates `pypdf` |
| 2 | `is_inside_math` used `start <= pos`, needlessly rejecting *safe* boundary splits | Off-by-one that degrades quality without failing anything |
| 3 | The Hebrew PDF fixture had no `hyperref`, so it silently exercised **none** of the outline-mapping code | A fixture that passes while testing nothing |
| 4 | The model emitted `[3]` / `[SOURCE 3]` instead of the label — numbered markers competed with citation syntax | Requires a real LLM to exhibit |
| 5 | PDF outline titles contained embedded newlines, splitting citation labels across lines and making them **invisible to the label parser** — silently breaking eval scoring | Real-world data shape; fixed with `normalize_title` in both parsers |
| 6 | `AnswerService` deferred abstention to the retriever's config, so the UI slider and the eval sweep **had no effect** | Both paths "worked"; only the values never changed |
| 7 | The default `abstain_threshold` of 0.35 was **inert** — see §6 | Requires real embedding score distributions |
| 8 | `IngestState` lives at `chroma_dir/ingest_state.json` — **one file shared by every collection**, with no record of which collection chunks landed in. A stale entry silently skips a file, leaving an invisible hole | Needs two collections and a wiped store to manifest |
| 9 | `DuplicateIDError` — content-derived IDs mean two byte-identical sections in different files share an ID. Chroma tolerates a repeated ID **across** upsert calls but rejects it **within** one call | No synthetic corpus contains two byte-identical sections |

Bug 8 remains a known wart; `--force` is the current workaround. Bug 9 is fixed by
collapsing duplicates within a batch in `ChromaStore.add`, keeping the first occurrence —
the fix belongs in the Chroma adapter, so the `VectorStore` contract stays clean and the
in-memory fake need not imitate a vendor quirk.

---

## 9. FAQ

**Why Chroma?**
File-based, no server process, runs on a laptop, and has the metadata filtering this
project needs. It sits behind the `VectorStore` Protocol, so the choice is reversible.

**Why not fine-tune the model on my notes?**
Because it would not do what you want. **Ingestion is not training.** Adding a course to the
store changes *what can be retrieved*; it never touches a single model weight. Fine-tuning
teaches style and format, not facts you can cite — and a fine-tuned model still cannot tell
you *which file* a claim came from, which is the entire point here. Retrieval also updates
in minutes and is trivially correctable; a fine-tune is neither.

**Why abstain instead of always answering?**
Because an uncited or wrongly-cited physics answer is worse than no answer — it costs more
time to falsify than it saves. Abstention is also cheap: it happens *before* the generator
is called, so a question outside the corpus returns in seconds.

**Why one collection per course rather than one big one?**
The abstention threshold is calibrated from the separation between answerable and
unanswerable confidence bands, and that separation is a property of the corpus. Mixing
courses blurs both bands into one middle that no single threshold splits well. See §5.

**How do I add a new course?**

```bash
rsync -a --include='*/' --include='*.tex' --include='*.pdf' --exclude='*' \
      ~/GoogleDrive/path/to/course/ ~/.cache/physics-rag/corpus/course/
rag ingest ~/.cache/physics-rag/corpus/course --collection newcourse
rag eval eval/newcourse.yaml --calibrate --collection newcourse   # derive its own threshold
```

Stop `llama-server` first. The threshold sweep is not optional if you intend to trust the
abstention gate on the new collection.

**How do I know the citations are real?**
Two layers. `--show-sources` prints the retrieved chunks with their scores and math
fidelity, so a citation can be checked against the text that produced it. And the eval
harness scores citation accuracy against an answer key — which is how we know it is
currently **0.500 on the demonstration set**, not 1.0. The number is published precisely
because it is not flattering.

**Why does Hebrew matter so much here?**
It eliminates the default embedder and forces a cross-lingual one, and it introduces bidi
control characters that would otherwise break content hashing. It is also the reason the
committed test fixtures include a Hebrew `.tex` *and* a Hebrew PDF — so the bilingual path
is genuinely covered rather than assumed.

**What happens to equations that come from PDFs?**
They are tagged `degraded` and the prompt forbids presenting them as exact LaTeX. Trust
equations from `.tex` sources. This is a mitigation, not a fix — see §10.

**Why not OCR the PDFs?**
Because the text layer is already there and extracts cleanly; OCR would be slower and add
its own errors. OCR does not solve the actual problem either — the loss is in `pdftotext`'s
flattening of *layout* (superscripts, fractions), and an OCR pass over a rendered page has
the same difficulty. A LaTeX-aware formula recognizer would be the real fix, and is out of
scope.

**Why is CI meaningful if it never runs the real model?**
It verifies the parts that are deterministic: parser structure extraction, math-boundary
safety, chunk metadata, the abstention decision, citation label parsing, and eval scoring.
It also enforces that the Protocol seams are real, since fakes must be substitutable. What
CI cannot catch is §8 — which is why that section exists.

---

## 10. Known limitations

- **`.lyx` files are not parsed.** One corpus year stores its notes only as `.lyx` (109
  `.lyx`, 0 `.tex`), so that year is reachable **only** through compiled PDFs, with degraded
  math. `lyx -e latex` conversion is the obvious extension.
- **PDF math is not trustworthy.** Chunks are tagged `degraded` and the prompt forbids
  presenting them as exact LaTeX, but a small model does not always comply. This is measured
  by the eval rather than assumed away.
- **Section-level citation requires PDF bookmarks.** Textbooks have them; lecture slide decks
  usually do not, and fall back to page-level citations.
- **Retrieval can return several chunks from one section.** Dedup removes only byte-identical
  text; diversity-aware reranking is not implemented.
- **`IngestState` is shared across collections** and does not record where chunks landed, so a
  stale entry can silently skip a file. `--force` is the workaround; the durable fix is to key
  state by collection.
- **No measured 25-item eval table exists**, and the committed questions are TOC-derived rather
  than owner-authored. See §6.
- **Full-corpus coverage is incomplete.** Only some course years are staged and ingested.

## 11. Next steps

Roughly in order of how much they would improve the project's credibility:

1. **Owner-authored eval pairs.** Replacing the TOC-derived questions with real coursework
   questions removes the circularity and makes the hit-rate mean something.
2. **Complete the 25-item calibrated run** and publish the measured table.
3. **Key `IngestState` by collection**, removing the silent-skip failure mode.
4. **Calibrate a `qm2`-specific threshold** once its ingest completes.
5. **`.lyx` support** via `lyx -e latex`, unlocking a whole corpus year.
6. **Diversity-aware reranking**, so `top_k` buys distinct sections rather than neighbours —
   which matters more than usual here, since the system is prompt-bound.

---

## Appendix: commands

```bash
uv sync                                  # --extra ui for the Gradio front-end
uv run pytest                            # no model, no corpus required
uv run ruff check .

# stage the corpus locally (FUSE mounts are too slow to ingest from directly)
rsync -a --exclude='Docs/' --exclude='.*/' \
      --include='*/' --include='*.tex' --include='*.pdf' --exclude='*' \
      ~/GoogleDrive/ ~/.cache/physics-rag/corpus/

rag ingest ~/.cache/physics-rag/corpus   # embedder only — stop llama-server first
rag stats

llm-serve rag-quality -d                 # llama-server on :8080
rag ask "What is the entropy of an Einstein solid?"
rag ask "..." --show-sources
rag eval eval/eval_set.yaml --calibrate
rag ui
```

Configuration is a TOML file passed with `-c`; unknown keys are rejected rather than
silently ignored:

```toml
[physics_rag]
collection_name = "qm2"
exclude_dirs = ["Docs", "_archive"]
top_k = 6
abstain_threshold = 0.82   # calibrate this per collection
```
