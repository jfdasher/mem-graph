from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from memgraph.rerank import Reranker
from memgraph.schema import Fact


def _make_fact(text: str, score: float = 0.5) -> Fact:
    return Fact(
        id=uuid4(),
        text=text,
        fact_type="world",
        source_chunk_id=uuid4(),
        temporal_start=None,
        temporal_end=None,
        confidence=0.8,
        tags=[],
        metadata={},
        created_at=datetime.now(UTC),
        score=score,
    )


def test_rerank_reorders_candidates_by_relevance() -> None:
    reranker = Reranker()
    query = "quantum physics experiments"
    facts = [
        _make_fact("The weather is sunny today.", score=0.9),
        _make_fact("Quantum entanglement experiments showed non-local effects.", score=0.1),
        _make_fact("I had coffee this morning.", score=0.7),
    ]
    result = reranker.rerank(query, facts, k=3)
    assert result[0].text == "Quantum entanglement experiments showed non-local effects."


def test_rerank_preserves_top_k_count() -> None:
    reranker = Reranker()
    facts = [_make_fact(f"fact number {i}") for i in range(20)]
    result = reranker.rerank("some query", facts, k=5)
    assert len(result) == 5
