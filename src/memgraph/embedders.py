from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    def initialize(self) -> None:  # noqa: B027
        """Load model / create client. Called lazily on first embed()."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder(Embedder):
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._dimension: int | None = None

    def initialize(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import (
            SentenceTransformer,  # type: ignore[import-untyped,unused-ignore]
        )

        self._model = SentenceTransformer(self.model_name, device="cpu")
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or getattr(self._model, "get_sentence_embedding_dimension", None)
        self._dimension = get_dim()  # type: ignore[misc]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self.initialize()
        if self._dimension is None:
            raise RuntimeError("dimension unavailable after initialize()")
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.initialize()
        assert self._model is not None
        result = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in result]


class OpenAIEmbedder(Embedder):
    MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any = None

    def initialize(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key)

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS[self.model]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.initialize()
        assert self._client is not None
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]
