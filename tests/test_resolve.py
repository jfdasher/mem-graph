from __future__ import annotations

from sqlalchemy import Engine, text

from memgraph.resolve import resolve_or_create


def _get_entity(engine: Engine, entity_id) -> dict:  # type: ignore[type-arg]
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, name, type, aliases FROM entities WHERE id = :id"),
            {"id": str(entity_id)},
        ).fetchone()
    assert row is not None
    return {"id": row.id, "name": row.name, "type": row.type, "aliases": list(row.aliases or [])}


def _entity_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM entities")).scalar_one()


def test_new_name_creates_new_entity(engine: Engine, clean_db) -> None:
    eid = resolve_or_create(engine, "Zelda Vance", "person")
    e = _get_entity(engine, eid)
    assert e["name"] == "Zelda Vance"
    assert e["type"] == "person"
    assert _entity_count(engine) == 1


def test_exact_match_returns_existing_entity(engine: Engine, clean_db) -> None:
    eid1 = resolve_or_create(engine, "Alice Smith", "person")
    eid2 = resolve_or_create(engine, "Alice Smith", "person")
    assert eid1 == eid2
    assert _entity_count(engine) == 1


def test_case_insensitive_match(engine: Engine, clean_db) -> None:
    eid1 = resolve_or_create(engine, "Alice Smith", "person")
    eid2 = resolve_or_create(engine, "alice smith", "person")
    assert eid1 == eid2
    assert _entity_count(engine) == 1


def test_trigram_matches_minor_typo(engine: Engine, clean_db) -> None:
    # "Montgomery Blackwod" vs "Montgomery Blackwood" has similarity ~0.86,
    # reliably above high_threshold=0.85.
    eid1 = resolve_or_create(engine, "Montgomery Blackwood", "person")
    eid2 = resolve_or_create(engine, "Montgomery Blackwod", "person")  # typo
    assert eid1 == eid2, "Trigram match should have found the canonical entity"
    assert _entity_count(engine) == 1


def test_trigram_threshold_configurable(engine: Engine, clean_db) -> None:
    eid1 = resolve_or_create(engine, "Alice Smith", "person")
    # Very dissimilar name with high threshold → creates a new entity
    eid2 = resolve_or_create(engine, "Bob Jones", "person", low_threshold=0.95)
    assert eid1 != eid2
    assert _entity_count(engine) == 2


def test_alias_match_returns_canonical_id(engine: Engine, clean_db) -> None:
    eid1 = resolve_or_create(engine, "Robert Johnson", "person")
    # Add an alias manually
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE entities SET aliases = ARRAY['Bob Johnson'] WHERE id = :id"),
            {"id": str(eid1)},
        )
        conn.commit()

    eid2 = resolve_or_create(engine, "Bob Johnson", "person")
    assert eid1 == eid2
    assert _entity_count(engine) == 1
