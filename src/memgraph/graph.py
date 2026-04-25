from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Engine, text


@dataclass(frozen=True)
class ExpandedEntity:
    id: UUID
    name: str
    type: str | None
    score: float  # hop_decay * co_count


def upsert_entity_links(engine: Engine, entity_ids: list[UUID]) -> None:
    """Upsert co-occurrence edges for all pairs in entity_ids.

    Maintains entity_a_id < entity_b_id via UUID string comparison.
    """
    if len(entity_ids) < 2:
        return

    pairs = [
        (min(str(a), str(b)), max(str(a), str(b)))
        for i, a in enumerate(entity_ids)
        for b in entity_ids[i + 1:]
    ]

    with engine.connect() as conn:
        for a_str, b_str in pairs:
            conn.execute(
                text("""
                    INSERT INTO entity_links (entity_a_id, entity_b_id, co_count, last_seen_at)
                    VALUES (:a, :b, 1, NOW())
                    ON CONFLICT (entity_a_id, entity_b_id)
                    DO UPDATE SET
                        co_count = entity_links.co_count + 1,
                        last_seen_at = NOW()
                """),
                {"a": a_str, "b": b_str},
            )
        conn.commit()


def expand(
    engine: Engine,
    seed_entity_ids: list[UUID],
    *,
    hops: int = 1,
) -> list[ExpandedEntity]:
    """Expand outward from seed entities via entity_links.

    Returns entities ranked by hop_decay * co_count (descending).
    hop_decay = 1.0 / hop (1.0 for hop=1, 0.5 for hop=2, etc.)
    """
    if not seed_entity_ids:
        return []

    seen: set[UUID] = set(seed_entity_ids)
    scores: dict[UUID, float] = {}
    current_frontier = list(seed_entity_ids)

    for hop in range(1, hops + 1):
        decay = 1.0 / hop
        next_frontier: list[UUID] = []

        seed_strs = [str(eid) for eid in current_frontier]
        seeds_pg = "{" + ",".join(seed_strs) + "}"
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        CASE
                            WHEN entity_a_id = ANY(CAST(:seeds AS uuid[]))
                            THEN entity_b_id
                            ELSE entity_a_id
                        END AS neighbor_id,
                        co_count
                    FROM entity_links
                    WHERE entity_a_id = ANY(CAST(:seeds AS uuid[]))
                       OR entity_b_id = ANY(CAST(:seeds AS uuid[]))
                """),
                {"seeds": seeds_pg},
            ).fetchall()

        for row in rows:
            neighbor_id = UUID(str(row.neighbor_id))
            if neighbor_id in seen:
                continue
            candidate_score = decay * row.co_count
            if neighbor_id not in scores or candidate_score > scores[neighbor_id]:
                scores[neighbor_id] = candidate_score
            next_frontier.append(neighbor_id)
            seen.add(neighbor_id)

        current_frontier = list(set(next_frontier))

    if not scores:
        return []

    ids_pg = "{" + ",".join(str(eid) for eid in scores) + "}"
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, name, type FROM entities WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": ids_pg},
        ).fetchall()

    result = [
        ExpandedEntity(
            id=UUID(str(row.id)),
            name=row.name,
            type=row.type,
            score=scores[UUID(str(row.id))],
        )
        for row in rows
    ]
    result.sort(key=lambda e: e.score, reverse=True)
    return result
