from __future__ import annotations

from sqlalchemy import Engine

from memgraph.ingest import ingest_chunk
from memgraph.llm import MockLLM
from memgraph.retrieve import retrieve_facts
from tests.fixtures.canned_llm import ALICE_BOB_COLAB, BOB_DEADLINE


def _setup_alice_bob_corpus(engine, embedder) -> None:
    ingest_chunk(engine, embedder, MockLLM([ALICE_BOB_COLAB]),
                 "Alice collaborated with Bob on the merger analysis.")
    ingest_chunk(engine, embedder, MockLLM([BOB_DEADLINE]),
                 "Bob's merger analysis deadline is next Friday.")


def test_semantic_mode_returns_facts(engine: Engine, local_embedder, clean_db) -> None:
    _setup_alice_bob_corpus(engine, local_embedder)
    results = retrieve_facts(engine, local_embedder, "merger analysis", mode="semantic", k=5)
    assert len(results) >= 1
    assert any("merger" in r.text.lower() for r in results)


def test_bm25_mode_returns_facts(engine: Engine, local_embedder, clean_db) -> None:
    _setup_alice_bob_corpus(engine, local_embedder)
    results = retrieve_facts(engine, local_embedder, "merger analysis", mode="bm25", k=5)
    assert len(results) >= 1
    assert any("merger" in r.text.lower() for r in results)


def test_hybrid_mode_returns_facts(engine: Engine, local_embedder, clean_db) -> None:
    _setup_alice_bob_corpus(engine, local_embedder)
    results = retrieve_facts(engine, local_embedder, "merger analysis", mode="hybrid", k=5)
    assert len(results) >= 1


def test_graph_catches_what_hybrid_misses(engine: Engine, local_embedder, clean_db) -> None:
    """Graph retrieval surfaces Bob's deadline when queried for 'Alice'."""
    _setup_alice_bob_corpus(engine, local_embedder)

    graph_results = retrieve_facts(engine, local_embedder, "Alice", mode="graph", k=10, hops=1)
    graph_texts = {r.text for r in graph_results}

    deadline_text = "Bob's merger analysis deadline is next Friday."
    assert deadline_text in graph_texts, (
        f"Graph mode should surface Bob's deadline via Alice->Bob link. "
        f"Got: {graph_texts}"
    )


def test_full_mode_returns_facts(engine: Engine, local_embedder, clean_db) -> None:
    _setup_alice_bob_corpus(engine, local_embedder)
    results = retrieve_facts(engine, local_embedder, "merger analysis", mode="full", k=5)
    assert len(results) >= 1
