from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

# intfloat/e5 models are trained with these asymmetric prefixes. Omitting them
# degrades retrieval silently - there is no error, only worse results - so they
# are applied inside the Embedder and never left to callers.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embedder:
    """Embedder backed by a sentence-transformers e5 model."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        # Imported lazily so this module can be imported without torch present.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        # get_sentence_embedding_dimension was renamed in recent releases.
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if callable(get_dimension):
            return int(get_dimension())
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) == 0:
            return []
        prefixed = [PASSAGE_PREFIX + text for text in texts]
        matrix = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(value) for value in row] for row in matrix]

    def embed_query(self, text: str) -> list[float]:
        prefixed = [QUERY_PREFIX + text]
        matrix = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(value) for value in matrix[0]]
