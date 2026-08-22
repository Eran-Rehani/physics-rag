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
abstain_threshold = 0.82
```

## Evaluation

The harness scores three things and calibrates a fourth:

- **retrieval hit-rate** — did any retrieved chunk come from an expected file?
- **citation accuracy** — do the emitted `[file, section/page]` labels match the expected source?
- **abstention precision / recall / F1** — measured against deliberately unanswerable questions
- **`--calibrate`** sweeps the abstain threshold and reports the value maximising abstention F1,
  asking each question once and recomputing decisions per threshold

`abstain_threshold` currently defaults to `0.82`, calibrated from the six-question demonstration
run below. It is **corpus-specific**: the confidence band shifts with corpus composition, so a
different collection needs its own sweep. The negative items exist so the threshold is derived
from observed separation rather than guessed.

### Measured: why the threshold must be calibrated

Running the harness against a 4216-chunk collection of astrophysics textbooks (6 demonstration
questions: 4 answerable, 2 deliberately out-of-corpus) produced:

| Metric | Result |
|---|---|
| Retrieval hit-rate | **1.000** (4/4) |
| Citation accuracy | 0.500 (2/4) |
| Abstention precision / recall / F1 | **1.000** |
| Mean confidence, answerable | 0.889 |
| Mean confidence, negative | 0.794 |
| **Calibrated `abstain_threshold`** | **0.82** |

The important result is the *separation*, not the averages. `multilingual-e5` cosine scores
compress into a narrow high band — an utterly unrelated question ("optimal cache eviction
policy for a distributed key-value store") still scored **0.801**. A plausible-looking default
of `0.35` is therefore **inert**: it never fires, and every question would reach the generator.
The current `0.82` default is derived from this demonstration measurement and is not the final
threshold for the coursework eval set.

Two layers produce the abstention, which is why the out-of-corpus questions were caught even
before calibration: the confidence gate, and the model's own instruction to reply
"not found in corpus". Citation accuracy of 0.5 is a real miss, not a rounding artifact —
on two questions the model cited a source outside the expected set.

> These six questions are a **demonstration** that the harness works end to end, not the
> project's final score.

### Status of the full eval set, stated plainly

`eval/eval_set.yaml` holds 20 answerable questions plus 5 deliberate negatives over the same
astrophysics collection. Two caveats matter more than the numbers will:

1. **The questions are TOC-derived, not owner-authored.** They were written from the PDFs'
   outline section paths rather than from coursework. Because each question was reverse-engineered
   from its own answer key, retrieval hit-rate on this set is **partly circular** — it measures
   that the pipeline retrieves what it was pointed at, not that it answers questions a student
   would actually ask. Treat it as a smoke test of the harness, not as a quality score.
2. **The 25-item run has not been completed.** A `--calibrate` run was paused partway; the
   harness reports only after scoring every item, so there are no partial numbers to publish.
   No measured coursework table appears here because one does not exist yet, and inventing one
   would defeat the point of the project.

A pre-flight against the store's metadata confirmed all 25 expected `(file, section)` pairs
resolve (`0` unmatched, en-dashes included), so any citation failure the run reports will be
attributable to retrieval or generation rather than to a mistyped answer key.

### Reproducing the measurement

Generation and embedding **cannot run concurrently on 16 GB** — `llama-server` holds ~4.8 GB
plus a 24k-context q8 KV cache, and adding the embedder drove available RAM low enough for the
kernel to kill an ingest mid-run. Serialise the two:

```bash
llm-serve rag-quality -d      # wait for {"status":"ok"} on :8080
uv run rag eval --calibrate   # ~140 s per answerable question
```

Prompt processing measures **~40 tok/s** at 2.2k-token contexts on this hardware, so the cost is
dominated by the retrieved context, not by the generated answer. Keep `top_k` small.

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
