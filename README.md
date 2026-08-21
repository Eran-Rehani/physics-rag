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
abstain_threshold = 0.35
```

## Evaluation

The harness scores three things and calibrates a fourth:

- **retrieval hit-rate** — did any retrieved chunk come from an expected file?
- **citation accuracy** — do the emitted `[file, section/page]` labels match the expected source?
- **abstention precision / recall / F1** — measured against deliberately unanswerable questions
- **`--calibrate`** sweeps the abstain threshold and reports the value maximising abstention F1,
  asking each question once and recomputing decisions per threshold

`abstain_threshold` defaults to `0.35`, which is a **placeholder**. It is meant to be replaced
by a measured value: the negatives in the eval set exist so the threshold can be derived from
the observed separation between answerable and unanswerable questions.

> **Eval results:** the eval set is authored from the corpus owner's own coursework and is not
> committed with content. Populate `eval/eval_set.yaml` (15–20 answerable pairs plus ~5
> negatives), run `rag eval --calibrate`, and paste the report here.

## Testing

```bash
uv run pytest        # 94 tests, no model and no corpus required
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
