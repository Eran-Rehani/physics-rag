"""Minimal Gradio front-end over the same AnswerService the CLI uses.

Kept deliberately thin: all behaviour (retrieval, abstention, citation binding)
lives in the library modules, so the UI is a view and nothing else.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from physics_rag.answer import AnswerService
from physics_rag.config import Config, default_config, from_toml


def build_service(config: Config, collection: str | None = None) -> AnswerService:
    from physics_rag.embeddings import E5Embedder
    from physics_rag.generation import LlamaServerGenerator
    from physics_rag.retrieval import Retriever
    from physics_rag.store import ChromaStore

    embedder = E5Embedder(config.embedding_model)
    store = ChromaStore(config.chroma_dir, collection or config.collection_name)
    retriever = Retriever(embedder, store, config)
    generator = LlamaServerGenerator(config.llama_server_url, timeout=config.generation_timeout)
    return AnswerService(retriever, generator, config)


def format_sources_markdown(answer_results: list) -> str:
    from physics_rag.retrieval import format_citation

    if not answer_results:
        return "_No sources retrieved._"
    lines = ["| score | citation | math |", "| --- | --- | --- |"]
    for result in answer_results:
        label = format_citation(result).replace("|", "\\|")
        lines.append(f"| {result.score:.3f} | {label} | {result.math_fidelity} |")
    return "\n".join(lines)


def build_demo(config: Config, collection: str | None = None):
    import gradio as gr

    service = build_service(config, collection)

    def answer_question(question: str, top_k: int, threshold: float):
        question = (question or "").strip()
        if not question:
            return "Ask a question first.", "", ""

        local_service = service.with_config(replace(config, abstain_threshold=threshold))
        answer = local_service.ask(question, top_k=int(top_k))

        status = (
            f"**abstained** (confidence {answer.confidence:.3f})"
            if answer.abstained
            else f"confidence {answer.confidence:.3f}"
        )
        return answer.text, status, format_sources_markdown(answer.results)

    with gr.Blocks(title="Physics RAG") as demo:
        gr.Markdown(
            "# Physics RAG\n"
            "Answers come only from your own notes and textbooks, with a citation per claim. "
            "When retrieval confidence is low the system says *not found in corpus* "
            "instead of guessing."
        )
        with gr.Row():
            question = gr.Textbox(
                label="Question (English)",
                placeholder="What is the entropy of an Einstein solid?",
                scale=4,
            )
            ask_button = gr.Button("Ask", variant="primary", scale=1)
        with gr.Row():
            top_k = gr.Slider(1, 20, value=config.top_k, step=1, label="top-k")
            threshold = gr.Slider(
                0.0, 1.0, value=config.abstain_threshold, step=0.01, label="abstain threshold"
            )

        answer_box = gr.Markdown(label="Answer")
        status_box = gr.Markdown()
        sources_box = gr.Markdown(label="Sources")

        inputs = [question, top_k, threshold]
        outputs = [answer_box, status_box, sources_box]
        ask_button.click(answer_question, inputs=inputs, outputs=outputs)
        question.submit(answer_question, inputs=inputs, outputs=outputs)

    return demo


def main(config_path: Path | None = None, collection: str | None = None) -> None:
    config = default_config() if config_path is None else from_toml(config_path)
    build_demo(config, collection).launch()


if __name__ == "__main__":
    main()
