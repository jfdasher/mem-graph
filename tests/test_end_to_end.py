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
    # Turn 1: Alice-only fact (no Bob)
    ALICE_ONLY = {
        "facts": [{
            "text": "Alice presented the analysis findings to the board.",
            "fact_type": "experience",
            "entities": [{"name": "Alice", "type": "person"}],
            "temporal_start": None, "temporal_end": None, "confidence": 0.9,
        }]
    }
    # Turn 2: Alice+Bob co-occurrence (builds the link)
    # Turn 3: Bob-only deadline fact (no Alice)

    for text_content, canned in [
        ("Alice presented findings.", ALICE_ONLY),
        ("Alice told me she finished the merger analysis.", ALICE_BOB_COLAB),
        ("Bob said the deadline for his part is next Friday.", BOB_DEADLINE),
    ]:
        ingest_chunk(engine, local_embedder, MockLLM([canned]), text_content)

    results = retrieve_facts(
        engine, local_embedder, "Alice presented findings", mode="graph", k=10, hops=1
    )
    result_texts = {r.text for r in results}

    # Bob's deadline fact should be reached via Alice→Bob link
    assert any("deadline" in t.lower() or "friday" in t.lower() for t in result_texts), \
        f"Graph mode should surface Bob's deadline via Alice→Bob link. Got: {result_texts}"


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

    from memgraph.llm import AnthropicLLM

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    llm = AnthropicLLM(api_key=api_key)
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
