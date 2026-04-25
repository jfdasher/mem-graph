from __future__ import annotations

from dataclasses import replace
from typing import Any

from .schema import Fact


class Reranker:
    MODEL = "BAAI/bge-reranker-base"

    def __init__(self, model: str = MODEL) -> None:
        self.model_name = model
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import (
                CrossEncoder,  # type: ignore[import-untyped,unused-ignore]
            )

            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, facts: list[Fact], *, k: int = 10) -> list[Fact]:
        """Score facts against query; return top-k sorted by descending score."""
        if not facts:
            return []
        self._load()
        assert self._model is not None
        pairs = [[query, f.text] for f in facts]
        scores = self._model.predict(pairs)
        scored = sorted(zip(scores, facts, strict=True), key=lambda x: x[0], reverse=True)
        return [replace(f, score=float(s)) for s, f in scored[:k]]


def rerank(query: str, facts: list[Fact], *, k: int = 10) -> list[Fact]:
    """Module-level convenience using a cached global Reranker."""
    return _GLOBAL.rerank(query, facts, k=k)


_GLOBAL = Reranker()
