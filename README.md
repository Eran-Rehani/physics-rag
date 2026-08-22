# physics-rag

A local, citation-grounded RAG system that answers physics questions from a personal corpus
of LaTeX notes and PDF textbooks — **on a CPU-only laptop, with no data leaving the machine.**

Every answer cites `[filename, section/page]` per claim, and the system says
**"not found in corpus"** rather than guessing when retrieval confidence is low.

```console
$ rag ask "What is the Friedmann equation in cosmology?"

The Friedmann equation binds the curvature of space ($\kappa$), the expansion rate
($a(t)$), and the energy density ($\epsilon$) of the universe
[Introduction to Cosmology-Barbara Ryden-2022.pdf, 4 Cosmic Dynamics > 4.2 The Friedmann Equation].

confidence 0.886
```

## Why this is not a generic RAG demo

The corpus is real coursework, and it broke several defaults worth naming:

| Reality | What it forced |
|---|---|
| Notes are **majority Hebrew** prose with English/LaTeX math | An English-only embedder (`bge-small-en`) is useless here. Uses `multilingual-e5-small`; measured cosine between an English query and its Hebrew passage is **0.814** |
| `.tex` math is verbatim, PDF math is **flattened** by extraction (`E 2 = p2 c2`) | Every chunk is tagged `math_fidelity: exact\|degraded`, and the model is forbidden from reproducing LaTeX from degraded sources |
| Equations must never be corrupted | Chunk boundaries are provably never placed inside `$…$`, `\[…\]`, or a math environment. Verified on the real corpus: **2260 sections, 130 sub-split, zero boundaries inside math, zero text lost** |
| Hebrew PDF text arrives wrapped in bidi controls (U+202A–U+202C) | A shared normalizer strips them, so identical content reached via `.tex` and via PDF hashes equal |
| Many PDFs are compiled outputs of a `.tex` beside them | Dedup on `(directory, stem)` prefers the source, so near-identical text does not crowd retrieval or inflate hit-rate |
| The corpus lives on a Google Drive FUSE mount where `find` **times out at 120s** | Ingest runs from a local staging copy and is incremental and resumable |

## Architecture

Retrieval, generation and storage sit behind `typing.Protocol` seams, so the LLM or the
vector store can be swapped without touching the rest — and CI runs the whole pipeline on
deterministic fakes, needing neither the 4.8GB model nor the private corpus.

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

| Module | Responsibility |
|---|---|
| `parsers/tex.py`, `parsers/pdf.py` | Structure-preserving parsing; sections, page numbers, math fidelity |
| `normalize.py` | Bidi stripping, NFC, title flattening, content hashing — shared by both parsers |
| `chunking.py` | Semantic-section chunking with math-safe sub-splitting |
| `embeddings.py` | `Embedder` protocol; `E5Embedder` applies e5 prefixes internally |
| `store.py` | `VectorStore` protocol; `ChromaStore` with explicit cosine space |
| `retrieval.py` | `Retriever`, confidence, citation formatting, dedup |
| `generation.py` | `Generator` protocol; `LlamaServerGenerator` over HTTP |
| `answer.py` | Cite-or-abstain policy, prompt assembly, citation binding |
| `evaluation.py` | Hit-rate, citation correctness, abstention F1, threshold calibration |
| `cli.py`, `ui.py` | `rag` commands and a minimal Gradio front-end |

## Setup

Requires Python 3.14 (3.12+ works), [`uv`](https://docs.astral.sh/uv/), and `poppler-utils`
for `pdftotext`.

```bash
git clone <this repo> && cd physics-rag
uv sync                 # add --extra ui for the Gradio front-end
```

Stage your corpus locally (reading directly off a network/FUSE mount is painfully slow):

```bash
rsync -a --exclude='Docs/' --exclude='.*/' \
      --include='*/' --include='*.tex' --include='*.pdf' --exclude='*' \
      ~/GoogleDrive/ ~/.cache/physics-rag/corpus/
```

Start the generation model (llama.cpp `llama-server`, OpenAI-compatible endpoint on :8080):

```bash
llm-serve rag-quality     # or: llama-server -m rag-quality_gemma4-e2b-q8.gguf -c 8192 -ngl 0 --jinja
```

## Usage

```bash
rag ingest ~/.cache/physics-rag/corpus   # parse, chunk, embed, store (incremental + resumable)
rag ask "What is the entropy of an Einstein solid?"
rag ask "..." --show-sources             # show retrieved chunks, scores, math fidelity
rag stats                                # collection name, persist dir, chunk count
rag eval eval/eval_set.yaml              # score the eval set
rag eval --calibrate                     # sweep and report the best abstain threshold
rag ui                                   # Gradio front-end
```

Configuration is a TOML file passed with `-c`:

```toml
[physics_rag]
chroma_dir = "~/.cache/physics-rag/chroma"
collection_name = "physics"
top_k = 6
abstain_threshold = 0.84   # calibrate this per collection
```

## Evaluation

The harness scores three things and calibrates a fourth:

- **retrieval hit-rate** — did any retrieved chunk come from an expected file?
- **citation accuracy** — do the emitted `[file, section/page]` labels match the expected source?
- **abstention precision / recall / F1** — measured against deliberately unanswerable questions
- **`--calibrate`** sweeps the abstain threshold and reports the value maximising abstention F1

### Measured: 25-item run

25 questions (20 answerable, 5 deliberately out-of-corpus) against the 4097-chunk
astrophysics collection, `top_k = 6`, `abstain_threshold = 0.82` (the value in force during the run):

| Metric | Result |
|---|---|
| Retrieval hit-rate | **0.900** (18/20) |
| Citation accuracy (of answered) | **0.368** (7/19) |
| Abstention precision | 0.833 |
| Abstention recall | **1.000** |
| Abstention F1 | 0.909 |
| Mean confidence, answerable | 0.882 |
| Mean confidence, negative | 0.814 |

**Retrieval works; citation does not.** The pipeline puts a chunk from the right file in
context 90% of the time, and then the model cites the right source in only 37% of the
answers it produces. The gap between those two numbers is the honest headline of this
project: retrieval is a solved sub-problem here, and *citation binding by a 2B-parameter
model is not*. A larger generator, or a constrained decoding scheme that only permits
emitting labels present in the context, is the obvious next lever — not more retrieval
tuning.

Since the eval set was reverse-engineered from these files' own tables of contents
(see below), 0.900 should be read as an upper bound on hit-rate, not a typical result.

### The threshold is too low, and the eval proves it

Every negative was correctly refused (recall 1.000), but **not by the mechanism that was
supposed to catch them**:

| Negative question | Confidence | Caught by |
|---|---|---|
| `neg-sqlite-wal` | 0.780 | confidence gate |
| `neg-http2-hpack` | 0.814 | confidence gate |
| `neg-qcd-string-tension` | 0.819 | confidence gate |
| `neg-kubernetes-scheduling` | **0.827** | the model's own refusal |
| `neg-transformer-rope` | **0.830** | the model's own refusal |

Two of five negatives scored *above* the 0.82 gate and reached the generator. They were
refused only because the prompt instructs the model to answer "not found in corpus" when
the context does not support an answer. That second layer is doing work the first layer
was supposed to do — and it is the less reliable of the two, since it depends on a small
model complying with an instruction.

The same second layer produced the one false abstention: `habitable-zone` scored 0.851,
passed the gate, and was then refused by the model anyway. That single error is the entire
difference between abstention precision 0.833 and 1.000.

The fix is visible directly in the data:

```
highest-scoring negative   0.830   (neg-transformer-rope)
lowest-scoring answerable  0.850   (hr-main-sequence)
```

The two classes are **perfectly separable** on this run — any threshold in `(0.830, 0.850]`
gates every negative correctly without rejecting a single answerable question. The `0.82`
default, inherited from a six-question demonstration, sat below that window and leaked two
negatives into the generator.

`--calibrate` swept 51 candidate thresholds and returned **0.84**, which is now the default
in `config.py`. On this eval set it lifts abstention precision from 0.833 to 1.000, since
both leaked negatives (0.827, 0.830) fall below it while the lowest answerable (0.850) stays
above.

This is the argument for calibration restated with better evidence: a threshold is a
property of a corpus and an embedder, and it is *measured*, never chosen. The previous value
was derived from six questions and was wrong by a small but consequential margin at 25 —
0.02 of cosine distance, which is two of five negatives.

### Why a naive threshold does not work at all

`multilingual-e5` cosine scores compress into a narrow high band. Across this run the
entire spread — from a question about **HTTP/2 HPACK header compression** to one about the
Friedmann equation — is `0.780` to `0.906`, a total range of 0.126. A plausible-looking
default of `0.35` is therefore not conservative but **inert**: it never fires, every
question reaches the generator, and the abstention feature silently does not exist.

The usable signal is not the absolute score but the *separation* between the classes:
0.882 mean answerable versus 0.814 mean negative.

### Provenance of the eval set, stated plainly

`eval/eval_set.yaml` holds 20 answerable questions plus 5 negatives. One caveat matters
more than any number above:

**The questions are TOC-derived, not owner-authored.** They were written from the PDFs'
outline section paths rather than from coursework. Because each question was
reverse-engineered from its own answer key, retrieval hit-rate on this set is **partly
circular** — it measures that the pipeline retrieves what it was pointed at, not that it
answers questions a student would actually ask. Treat 0.900 as a smoke test of the
harness, not as a quality score. Replacing these with real coursework questions is the
top open item.

Citation accuracy is *less* affected by this circularity, and is the more trustworthy of
the two numbers: the model receives the same context either way, and the question is only
whether it attributes its claims correctly.

A pre-flight against the store's metadata confirmed all 25 expected `(file, section)` pairs
resolve (`0` unmatched, en-dashes included), so the citation failures above are attributable
to the model rather than to a mistyped answer key.

### Reproducing the measurement

Generation and embedding **cannot run concurrently on 16 GB** — `llama-server` holds ~4.8 GB
plus a 24k-context q8 KV cache, and adding the embedder during a bulk ingest drove available
RAM low enough for the kernel to kill it. Serialise the two:

```bash
llm-serve rag-quality -d      # wait for {"status":"ok"} on :8080
uv run rag eval --calibrate   # ~140 s per answerable question
```

The 25-item run above took **56 minutes**. Note that `--calibrate` performs a *second* full
pass over the eval set rather than reusing the confidences `run_eval` already computed,
roughly doubling the wall time; reusing them is a straightforward improvement.

Prompt processing measures **~40 tok/s** at 2.2k-token contexts on this hardware, so the cost
is dominated by the retrieved context, not by the generated answer. Keep `top_k` small.

## Testing

```bash
uv run pytest        # 93 tests, no model and no corpus required
uv run ruff check .
```

CI runs the suite on every push (`.github/workflows/tests.yml`). Because the corpus is private
and the model is 4.8GB, CI exercises the pipeline through the Protocol seams using
deterministic fakes. Parser tests run against small committed fixtures, including a Hebrew
`.tex` file and a Hebrew PDF, so the bilingual path is genuinely covered rather than assumed.

## Known limitations

- **`.lyx` files are not parsed.** One corpus year stores its notes only as `.lyx`, so that
  year is reachable only through its compiled PDFs (with degraded math). `lyx -e latex`
  conversion is the obvious extension.
- **PDF math is not trustworthy.** `pdftotext` flattens sub/superscripts. Chunks are tagged
  `degraded` and the prompt forbids presenting them as exact LaTeX, but a small model does not
  always comply — this is measured by the eval rather than assumed away.
- **Section-level citation needs PDF bookmarks.** Textbooks have them; lecture slide decks
  usually do not, and fall back to page-level citations.
- **Retrieval can return several chunks from one section.** Dedup only removes byte-identical
  text; diversity-aware reranking is not implemented.
- **Full-corpus ingest is slow.** Measured at ~116 ms/chunk on an i7-12700H; 11 textbook-heavy
  PDFs produced 4216 chunks in 490s.

## License

MIT
