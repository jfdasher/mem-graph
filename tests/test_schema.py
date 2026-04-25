from sqlalchemy import Engine, inspect, text


def _table_names(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def _index_names(engine: Engine, table: str) -> set[str]:
    return {ix["name"] for ix in inspect(engine).get_indexes(table)}


def test_graph_tables_exist(engine: Engine) -> None:
    tables = _table_names(engine)
    for t in ("facts", "entities", "fact_entities", "entity_links"):
        assert t in tables, f"Table {t!r} missing"


def test_facts_columns(engine: Engine) -> None:
    cols = {c["name"] for c in inspect(engine).get_columns("facts")}
    for col in ("id", "text", "fact_type", "source_chunk_id", "temporal_start",
                "temporal_end", "confidence", "embedding", "tsv", "tags",
                "metadata", "created_at"):
        assert col in cols, f"Column facts.{col} missing"


def test_facts_fk_to_chunks(engine: Engine) -> None:
    fks = inspect(engine).get_foreign_keys("facts")
    referred = {fk["referred_table"] for fk in fks}
    assert "chunks" in referred


def test_entities_trigram_index(engine: Engine) -> None:
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'entities'
                  AND indexdef LIKE '%gin_trgm_ops%'
            """)
        ).fetchone()
    assert row is not None, "pg_trgm GIN index missing on entities.name"


def test_entity_links_pk_ordering_constraint(engine: Engine) -> None:
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT pg_get_constraintdef(oid)
                FROM pg_constraint
                WHERE conrelid = 'entity_links'::regclass
                  AND contype = 'c'
            """)
        ).fetchone()
    assert row is not None, "entity_links CHECK constraint missing"
    assert "entity_a_id < entity_b_id" in row[0], (
        f"Expected ordering predicate in CHECK constraint, got: {row[0]}"
    )
