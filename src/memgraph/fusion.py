from __future__ import annotations

from uuid import UUID


def rrf(ranked_lists: list[list[UUID]], k: int = 60) -> dict[UUID, float]:
    """Reciprocal Rank Fusion over multiple ranked ID lists.

    Each list contributes 1 / (k + rank) per document, where rank is 1-based.
    Documents absent from a list contribute 0 from that list.
    Returns a mapping of document ID to total RRF score.
    """
    scores: dict[UUID, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
