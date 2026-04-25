from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import Engine, text

from memgraph.ingest import ingest_chunk
from memgraph.llm import MockLLM
from tests.fixtures.canned_llm import ALICE_BOB_COLAB


def _link_between(engine: Engine, eid_a: UUID, eid_b: UUID) -> dict | None:
    a, b = (eid_a, eid_b) if str(eid_a) < str(eid_b) else (eid_b, eid_a)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT co_count FROM entity_links "
                "WHERE entity_a_id = :a AND entity_b_id = :b"
            ),
            {"a": str(a), "b": str(b)},
        ).fetchone()
    return {"co_count": row.co_count} if row else None


def test_link_created_on_first_cooccurrence(engine: Engine, local_embedder, clean_db) -> None:
    ingest_chunk(engine, local_embedder, MockLLM([ALICE_BOB_COLAB]), "Alice collaborated with Bob.")

    with engine.connect() as conn:
        alice_id = conn.execute(
            text("SELECT id FROM entities WHERE LOWER(name) = 'alice'")
        ).scalar_one()
        bob_id = conn.execute(
            text("SELECT id FROM entities WHERE LOWER(name) = 'bob'")
        ).scalar_one()

    link = _link_between(engine, UUID(str(alice_id)), UUID(str(bob_id)))
    assert link is not None
    assert link["co_count"] == 1


def test_link_weight_increments_on_repeated_cooccurrence(
    engine: Engine, local_embedder, clean_db
) -> None:
    ingest_chunk(engine, local_embedder, MockLLM([ALICE_BOB_COLAB]), "Alice and Bob worked (1).")
    ingest_chunk(engine, local_embedder, MockLLM([ALICE_BOB_COLAB]), "Alice and Bob worked (2).")

    with engine.connect() as conn:
        alice_id = conn.execute(
            text("SELECT id FROM entities WHERE LOWER(name) = 'alice'")
        ).scalar_one()
        bob_id = conn.execute(
            text("SELECT id FROM entities WHERE LOWER(name) = 'bob'")
        ).scalar_one()

    link = _link_between(engine, UUID(str(alice_id)), UUID(str(bob_id)))
    assert link is not None
    assert link["co_count"] == 2


def test_expand_one_hop_returns_direct_neighbors(
    engine: Engine, local_embedder, clean_db
) -> None:
    from memgraph.graph import expand

    ingest_chunk(engine, local_embedder, MockLLM([ALICE_BOB_COLAB]), "Alice and Bob worked.")

    with engine.connect() as conn:
        alice_id = UUID(str(conn.execute(
            text("SELECT id FROM entities WHERE LOWER(name) = 'alice'")
        ).scalar_one()))

    neighbors = expand(engine, [alice_id], hops=1)
    neighbor_ids = {n.id for n in neighbors}

    with engine.connect() as conn:
        bob_id = UUID(str(conn.execute(
            text("SELECT id FROM entities WHERE LOWER(name) = 'bob'")
        ).scalar_one()))

    assert bob_id in neighbor_ids


def test_expand_two_hops_returns_neighbors_of_neighbors(
    engine: Engine, local_embedder, clean_db
) -> None:
    from memgraph.graph import expand

    bob_carol = {
        "facts": [{
            "text": "Bob mentors Carol on the project.",
            "fact_type": "experience",
            "entities": [{"name": "Bob", "type": "person"}, {"name": "Carol", "type": "person"}],
            "temporal_start": None, "temporal_end": None, "confidence": 0.85,
        }]
    }

    ingest_chunk(engine, local_embedder, MockLLM([ALICE_BOB_COLAB]), "Alice and Bob.")
    ingest_chunk(engine, local_embedder, MockLLM([bob_carol]), "Bob and Carol.")

    with engine.connect() as conn:
        alice_id = UUID(str(conn.execute(
            text("SELECT id FROM entities WHERE LOWER(name) = 'alice'")
        ).scalar_one()))

    neighbors_2hop = expand(engine, [alice_id], hops=2)
    neighbor_ids_2hop = {n.id for n in neighbors_2hop}

    with engine.connect() as conn:
        carol_id = UUID(str(conn.execute(
            text("SELECT id FROM entities WHERE LOWER(name) = 'carol'")
        ).scalar_one()))

    assert carol_id in neighbor_ids_2hop


def test_expand_multi_seed_picks_max_co_count(
    engine: Engine, local_embedder, clean_db
) -> None:
    """Neighbor reachable from two seeds with different co_counts gets max score."""
    from memgraph.graph import expand, upsert_entity_links
    from memgraph.resolve import resolve_or_create

    alice_id = resolve_or_create(engine, "Alice", "person")
    bob_id = resolve_or_create(engine, "Bob", "person")
    carol_id = resolve_or_create(engine, "Carol", "person")
    dave_id = resolve_or_create(engine, "Dave", "person")

    # Alice-Dave co_count=1, Bob-Dave co_count=3: max score from Dave's perspective is 3.0
    upsert_entity_links(engine, [alice_id, dave_id])  # co_count=1
    upsert_entity_links(engine, [bob_id, dave_id])
    upsert_entity_links(engine, [bob_id, dave_id])
    upsert_entity_links(engine, [bob_id, dave_id])   # co_count=3
    # Carol-Dave co_count=2 (intermediate)
    upsert_entity_links(engine, [carol_id, dave_id])
    upsert_entity_links(engine, [carol_id, dave_id])  # co_count=2

    # Seed on Alice+Bob+Carol simultaneously: Dave is reachable from all three
    neighbors = expand(engine, [alice_id, bob_id, carol_id], hops=1)
    dave = next((n for n in neighbors if n.name.lower() == "dave"), None)
    assert dave is not None, "Dave should be found as a neighbor"
    # hop_decay=1.0, max co_count=3 → max score=3.0 (from Bob→Dave edge)
    assert dave.score == pytest.approx(3.0), (
        f"Dave's score should be 3.0 (max co_count from Bob), got {dave.score}"
    )


def test_expand_ranking_uses_hop_decay_and_weight(
    engine: Engine, local_embedder, clean_db
) -> None:
    from memgraph.graph import expand

    # Ingest Alice-Bob twice (co_count=2) and Alice-Carol once (co_count=1)
    ingest_chunk(engine, local_embedder, MockLLM([ALICE_BOB_COLAB]), "Alice and Bob (1).")
    ingest_chunk(engine, local_embedder, MockLLM([ALICE_BOB_COLAB]), "Alice and Bob (2).")
    carol_data = {
        "facts": [{
            "text": "Alice met Carol briefly.",
            "fact_type": "experience",
            "entities": [{"name": "Alice", "type": "person"}, {"name": "Carol", "type": "person"}],
            "temporal_start": None, "temporal_end": None, "confidence": 0.8,
        }]
    }
    ingest_chunk(engine, local_embedder, MockLLM([carol_data]), "Alice and Carol.")

    with engine.connect() as conn:
        alice_id = UUID(str(conn.execute(
            text("SELECT id FROM entities WHERE LOWER(name) = 'alice'")
        ).scalar_one()))

    neighbors = expand(engine, [alice_id], hops=1)
    bob_pos = next((i for i, n in enumerate(neighbors) if n.name.lower() == "bob"), None)
    carol_pos = next((i for i, n in enumerate(neighbors) if n.name.lower() == "carol"), None)
    assert bob_pos is not None and carol_pos is not None
    assert bob_pos < carol_pos, f"Bob ({bob_pos}) should rank before Carol ({carol_pos})"
