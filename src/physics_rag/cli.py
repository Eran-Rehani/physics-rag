from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from physics_rag.config import Config, default_config, from_toml

app = typer.Typer(
    help="Local, citation-grounded RAG over your physics notes.", no_args_is_help=True
)
console = Console()


def _load_config(config_path: Path | None) -> Config:
    return default_config() if config_path is None else from_toml(config_path)


@app.command()
def ingest(
    path: Path | None = typer.Argument(None, help="Directory to ingest (default: staging_dir)."),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="TOML config file."),
    force: bool = typer.Option(False, "--force", help="Re-ingest even unchanged files."),
    batch_size: int = typer.Option(64, "--batch-size", help="Embedding batch size."),
    collection: str | None = typer.Option(None, "--collection", help="Chroma collection name."),
) -> None:
    """Parse, chunk, embed and store every .tex/.pdf under PATH."""
    # Heavy imports live here so --help stays fast and needs no torch.
    from physics_rag.embeddings import E5Embedder
    from physics_rag.ingest import IngestState
    from physics_rag.ingest import ingest as run_ingest
    from physics_rag.store import ChromaStore

    config = _load_config(config_path)
    collection_name = collection or config.collection_name
    target = path or config.staging_dir

    console.print(f"Ingesting [bold]{target}[/bold] into collection [bold]{collection_name}[/bold]")

    embedder = E5Embedder(config.embedding_model)
    store = ChromaStore(config.chroma_dir, collection_name)
    state = IngestState(config.chroma_dir / "ingest_state.json")

    stats = run_ingest(
        target,
        config=config,
        embedder=embedder,
        store=store,
        state=state,
        force=force,
        progress=lambda message: console.print(message, highlight=False),
        embed_batch_size=batch_size,
    )

    table = Table(title="Ingest complete")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for name, value in (
        ("Files seen", stats.files_seen),
        ("Files parsed", stats.files_parsed),
        ("Skipped unchanged", stats.files_skipped_unchanged),
        ("Skipped duplicate", stats.files_skipped_duplicate),
        ("Failed", stats.files_failed),
        ("Chunks added", stats.chunks_added),
        ("Chunks in store", store.count()),
    ):
        table.add_row(name, str(value))
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer, in English."),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="TOML config file."),
    top_k: int | None = typer.Option(None, "--top-k", "-k", help="Chunks to retrieve."),
    collection: str | None = typer.Option(None, "--collection", help="Chroma collection name."),
    show_sources: bool = typer.Option(
        False, "--show-sources", "-s", help="Print retrieved chunks."
    ),
    threshold: float | None = typer.Option(None, "--threshold", help="Override abstain threshold."),
) -> None:
    """Answer a question with citations, or abstain when confidence is low."""
    from physics_rag.answer import AnswerService
    from physics_rag.embeddings import E5Embedder
    from physics_rag.generation import GenerationError, LlamaServerGenerator
    from physics_rag.retrieval import Retriever, format_citation
    from physics_rag.store import ChromaStore

    config = _load_config(config_path)
    if threshold is not None:
        config = replace(config, abstain_threshold=threshold)
    collection_name = collection or config.collection_name

    embedder = E5Embedder(config.embedding_model)
    store = ChromaStore(config.chroma_dir, collection_name)
    retriever = Retriever(embedder, store, config)
    generator = LlamaServerGenerator(config.llama_server_url, timeout=config.generation_timeout)

    try:
        if not generator.health():
            console.print("[red]Generation server is not reachable.[/red]")
            console.print("Start it with:  llm-serve rag-quality")
            raise typer.Exit(1)
        answer = AnswerService(retriever, generator, config).ask(question, top_k=top_k)
    except GenerationError as exc:
        console.print(f"[red]Generation failed:[/red] {exc}")
        console.print("Start the server with:  llm-serve rag-quality")
        raise typer.Exit(1) from exc
    finally:
        generator.close()

    console.print()
    console.print(answer.text, highlight=False)
    console.print()
    console.print(f"[dim]confidence {answer.confidence:.3f}[/dim]")

    if show_sources:
        table = Table(title="Retrieved sources")
        table.add_column("Score", justify="right")
        table.add_column("Citation")
        table.add_column("Math")
        for result in answer.results:
            table.add_row(f"{result.score:.3f}", format_citation(result), result.math_fidelity)
        console.print(table)


@app.command()
def stats(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="TOML config file."),
    collection: str | None = typer.Option(None, "--collection", help="Chroma collection name."),
) -> None:
    """Show the collection name, persist directory and chunk count."""
    from physics_rag.store import ChromaStore

    config = _load_config(config_path)
    collection_name = collection or config.collection_name
    store = ChromaStore(config.chroma_dir, collection_name)

    console.print(f"Collection : {collection_name}")
    console.print(f"Persist dir: {config.chroma_dir}")
    console.print(f"Chunks     : {store.count()}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
