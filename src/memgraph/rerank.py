from __future__ import annotations

from .schema import Fact


def rerank(query: str, facts: list[Fact], *, k: int = 10) -> list[Fact]:
    return facts[:k]  # stub; replaced in Task 7
