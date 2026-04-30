from __future__ import annotations

import pytest
from sqlalchemy import Engine, text

from memgraph.ingest import ingest_chunk
from memgraph.llm import MockLLM
from memgraph.retrieve import retrieve_facts
from tests.fixtures.canned_llm import ALICE_BOB_COLAB, BOB_DEADLINE, SIMPLE_WORLD

THREE_TURN_TRANSCRIPT = [
    ("Alice told me she finished the merger analysis.", ALICE_BOB_COLAB),
    ("Bob said the deadline for his part is next Friday.", BOB_DEADLINE),
    ("The sky is blue and the weather is clear.", SIMPLE_WORLD),
]


def test_ingest_extracts_facts_and_builds_graph(engine: Engine, local_embedder, clean_db) -> None:
    for text_content, canned in THREE_TURN_TRANSCRIPT:
        ingest_chunk(engine, local_embedder, MockLLM([canned]), text_content)

    with engine.connect() as conn:
        fact_count = conn.execute(text("SELECT COUNT(*) FROM facts")).scalar_one()
        entity_count = conn.execute(text("SELECT COUNT(*) FROM entities")).scalar_one()
        link_count = conn.execute(text("SELECT COUNT(*) FROM entity_links")).scalar_one()

    assert fact_count >= 3, f"Expected ≥3 facts, got {fact_count}"
    assert entity_count >= 2, f"Expected ≥2 entities (Alice, Bob), got {entity_count}"
    assert link_count >= 1, f"Expected ≥1 entity link (Alice–Bob), got {link_count}"


def test_semantic_retrieval_surfaces_relevant_facts(
    engine: Engine, local_embedder, clean_db
) -> None:
    for text_content, canned in THREE_TURN_TRANSCRIPT:
        ingest_chunk(engine, local_embedder, MockLLM([canned]), text_content)

    results = retrieve_facts(engine, local_embedder, "merger analysis", mode="semantic", k=5)
    texts = [r.text for r in results]
    assert any("merger" in r.text.lower() for r in results), (
        f"Semantic query for 'merger analysis' should find merger facts. Got: {texts}"
    )


def test_graph_retrieval_crosses_entity_boundary(
    engine: Engine, local_embedder, clean_db
) -> None:
    """Graph mode must traverse Alice→Bob entity link to surface Bob's deadline.

    Corpus: 3 Alice-only facts fill the k=1 seed window (candidate_k=3), plus
    1 Bob-only deadline fact outside the window. The Alice→Bob link is inserted
    directly (no joint fact), so the deadline can only appear via expansion.
    Counterfactual: deleting the link removes Bob from expansion and the deadline
    disappears from the top-1 result.
    """
    from memgraph.graph import upsert_entity_links
    from memgraph.resolve import resolve_or_create

    alice_canned = [
        {"facts": [{"text": "Alice presented the analysis findings to the board.",
                    "fact_type": "experience",
                    "entities": [{"name": "Alice", "type": "person"}],
                    "temporal_start": None, "temporal_end": None, "confidence": 0.9}]},
        {"facts": [{"text": "Alice discussed her research conclusions with stakeholders.",
                    "fact_type": "experience",
                    "entities": [{"name": "Alice", "type": "person"}],
                    "temporal_start": None, "temporal_end": None, "confidence": 0.9}]},
        {"facts": [{"text": "Alice shared the preliminary results with her colleagues.",
                    "fact_type": "experience",
                    "entities": [{"name": "Alice", "type": "person"}],
                    "temporal_start": None, "temporal_end": None, "confidence": 0.9}]},
    ]
    for canned in alice_canned:
        ingest_chunk(engine, local_embedder, MockLLM([canned]), canned["facts"][0]["text"])

    # Bob's deadline fact — no Alice entity, so it only appears via graph expansion
    ingest_chunk(engine, local_embedder, MockLLM([BOB_DEADLINE]),
                 "Bob said the deadline for his part is next Friday.")

    # Insert the Alice→Bob co-occurrence link directly (no joint fact in corpus)
    alice_id = resolve_or_create(engine, "Alice", "person")
    bob_id = resolve_or_create(engine, "Bob", "person")
    upsert_entity_links(engine, [alice_id, bob_id])

    # k=1 → candidate_k=3: top-3 hybrid seeds are the Alice facts (closest to query).
    # Expansion reaches Bob (score=1.0) → BOB_DEADLINE scores 1.0, ranking above seeds.
    results = retrieve_facts(
        engine, local_embedder, "Alice presented findings", mode="graph", k=1, hops=1
    )
    result_texts = {r.text for r in results}
    assert any("deadline" in t.lower() or "friday" in t.lower() for t in result_texts), \
        f"Graph mode should surface Bob's deadline via Alice→Bob link. Got: {result_texts}"

    # Counterfactual: delete the link — Bob not reachable, deadline excluded
    with engine.connect() as conn:
        conn.execute(
            text("""
                DELETE FROM entity_links
                WHERE (entity_a_id = :a AND entity_b_id = :b)
                   OR (entity_a_id = :b AND entity_b_id = :a)
            """),
            {"a": str(alice_id), "b": str(bob_id)},
        )
        conn.commit()

    results_no_link = retrieve_facts(
        engine, local_embedder, "Alice presented findings", mode="graph", k=1, hops=1
    )
    result_texts_no_link = {r.text for r in results_no_link}
    has_deadline = any(
        "deadline" in t.lower() or "friday" in t.lower() for t in result_texts_no_link
    )
    assert not has_deadline, (
        f"After deleting Alice→Bob link, deadline should NOT appear in top-1 result. "
        f"Got: {result_texts_no_link}"
    )


def test_full_mode_returns_ranked_results(engine: Engine, local_embedder, clean_db) -> None:
    for text_content, canned in THREE_TURN_TRANSCRIPT:
        ingest_chunk(engine, local_embedder, MockLLM([canned]), text_content)

    results = retrieve_facts(engine, local_embedder, "merger deadline", mode="full", k=5)
    assert len(results) >= 1
    # Top result should be merger/deadline related
    top_texts = [r.text for r in results[:2]]
    assert any(
        "merger" in r.text.lower() or "deadline" in r.text.lower() or "friday" in r.text.lower()
        for r in results[:2]
    ), f"Top full-mode results should be merger/deadline related. Got: {top_texts}"


@pytest.mark.llm
def test_openai_extraction_produces_facts(engine: Engine, local_embedder, clean_db) -> None:
    import os

    from memgraph.llm import OpenAILLM

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    llm = OpenAILLM(api_key=api_key)
    chunk_id = ingest_chunk(
        engine, local_embedder, llm,
        "Alice finished the merger analysis and presented it to the board on Monday."
    )
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM facts WHERE source_chunk_id = :id"),
            {"id": str(chunk_id)},
        ).scalar_one()
    assert count >= 1, "OpenAI LLM should extract at least one fact"


@pytest.mark.llm
def test_anthropic_extraction_produces_facts(engine: Engine, local_embedder, clean_db) -> None:
    import os

    from memgraph.llm import OpenAILLM

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    llm = OpenAILLM(
        api_key=api_key,
        model="claude-haiku-4-5-20251001",
        base_url="https://api.anthropic.com/v1/",
    )
    chunk_id = ingest_chunk(
        engine, local_embedder, llm,
        "Bob completed the financial model and delivered it to Alice before the deadline."
    )
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM facts WHERE source_chunk_id = :id"),
            {"id": str(chunk_id)},
        ).scalar_one()
    assert count >= 1, "Anthropic LLM should extract at least one fact"
