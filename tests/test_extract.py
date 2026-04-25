from __future__ import annotations

from sqlalchemy import Engine, text

from memgraph.extract import extract_facts_from_chunk
from memgraph.llm import MockLLM
from tests.fixtures.canned_llm import EMPTY, EXPERIENCE_FACT, MALFORMED, SIMPLE_WORLD


def _fact_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM facts")).scalar_one()


def _insert_chunk(engine: Engine, text_val: str = "test") -> object:
    with engine.connect() as conn:
        chunk_id = conn.execute(
            text(
                "INSERT INTO chunks (text, embedding) "
                "VALUES (:t, (:e)::vector) RETURNING id"
            ),
            {"t": text_val, "e": "[" + ",".join(["0.0"] * 384) + "]"},
        ).scalar_one()
        conn.commit()
    return chunk_id


def test_extract_parses_facts_from_chunk(engine: Engine, local_embedder, clean_db) -> None:
    chunk_id = _insert_chunk(engine)
    llm = MockLLM([SIMPLE_WORLD])
    fact_ids = extract_facts_from_chunk(engine, local_embedder, llm, chunk_id, "The sky is blue.")

    assert len(fact_ids) == 1
    assert _fact_count(engine) == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT text, fact_type FROM facts WHERE id = :id"),
            {"id": str(fact_ids[0])},
        ).fetchone()
    assert row is not None
    assert row.text == "The sky is blue."
    assert row.fact_type == "world"


def test_extract_classifies_world_vs_experience(engine: Engine, local_embedder, clean_db) -> None:
    chunk_id = _insert_chunk(engine)
    llm = MockLLM([EXPERIENCE_FACT])
    fact_ids = extract_facts_from_chunk(
        engine, local_embedder, llm, chunk_id, "Alice got promoted."
    )

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT fact_type FROM facts WHERE id = :id"),
            {"id": str(fact_ids[0])},
        ).fetchone()
    assert row is not None
    assert row.fact_type == "experience"


def test_extract_returns_empty_on_empty_chunk(engine: Engine, local_embedder, clean_db) -> None:
    chunk_id = _insert_chunk(engine, "empty")
    llm = MockLLM([EMPTY])
    fact_ids = extract_facts_from_chunk(engine, local_embedder, llm, chunk_id, "")
    assert fact_ids == []
    assert _fact_count(engine) == 0


def test_extract_handles_malformed_llm_output_gracefully(
    engine: Engine, local_embedder, clean_db
) -> None:
    chunk_id = _insert_chunk(engine, "bad")
    llm = MockLLM([MALFORMED])
    fact_ids = extract_facts_from_chunk(engine, local_embedder, llm, chunk_id, "anything")
    assert fact_ids == []
