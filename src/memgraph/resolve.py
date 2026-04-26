from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

HIGH_THRESHOLD = 0.85
LOW_THRESHOLD = 0.70


def resolve_or_create(
    engine: Engine,
    name: str,
    entity_type: str | None,
    *,
    high_threshold: float = HIGH_THRESHOLD,
    low_threshold: float = LOW_THRESHOLD,
) -> UUID:
    """Return canonical UUID for an entity name, creating if not found.

    Lookup order:
    1. Exact alias match (name in entity.aliases array).
    2. Trigram similarity on entity.name: >= high_threshold -> reuse;
       >= low_threshold -> new entity + log near-miss; else -> new entity.
    """
    logger.debug("resolve(%r, type=%r)", name, entity_type)
    alias_id = _find_by_alias(engine, name)
    if alias_id is not None:
        logger.debug("resolve: alias match %r -> %s", name, alias_id)
        return alias_id

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, name, similarity(name, :name) AS sim
                FROM entities
                WHERE similarity(name, :name) > :low
                ORDER BY sim DESC
                LIMIT 5
            """),
            {"name": name, "low": low_threshold},
        ).fetchall()

    if rows:
        top = rows[0]
        if top.sim >= high_threshold:
            logger.debug("resolve: exact match %r -> %s (sim=%.3f)", name, top.id, top.sim)
            return UUID(str(top.id))
        logger.debug("Near-miss: %r <-> %r (sim=%.3f)", name, top.name, top.sim)

    new_id = _create_entity(engine, name, entity_type)
    logger.debug("resolve: created entity %r (%s) -> %s", name, entity_type, new_id)
    return new_id


def _find_by_alias(engine: Engine, name: str) -> UUID | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT id FROM entities
                WHERE :name = ANY(aliases)
                LIMIT 1
            """),
            {"name": name},
        ).fetchone()
    return UUID(str(row[0])) if row else None


def _create_entity(engine: Engine, name: str, entity_type: str | None) -> UUID:
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                INSERT INTO entities (name, type)
                VALUES (:name, :type)
                ON CONFLICT (LOWER(name)) DO NOTHING
                RETURNING id
            """),
            {"name": name, "type": entity_type},
        ).fetchone()

        if row is None:
            # Conflict on LOWER(name): fetch the existing entity
            row = conn.execute(
                text("SELECT id FROM entities WHERE LOWER(name) = LOWER(:name)"),
                {"name": name},
            ).fetchone()
            assert row is not None

        conn.commit()

    return UUID(str(row[0]))
