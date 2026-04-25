from __future__ import annotations

from dataclasses import replace
from typing import Literal
from uuid import UUID

from sqlalchemy import Engine, text

from .embedders import Embedder
from .fusion import rrf
from .graph import expand
from .schema import Fact, row_to_fact


def _emb_str(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _ids_pg(ids: list[str]) -> str:
    return "{" + ",".join(ids) + "}"


def retrieve_facts(
    engine: Engine,
    embedder: Embedder,
    query: str,
    *,
    mode: Literal["semantic", "bm25", "hybrid", "graph", "full"] = "hybrid",
    k: int = 10,
    hops: int = 1,
) -> list[Fact]:
    if mode == "semantic":
        return _semantic(engine, embedder, query, k=k)
    if mode == "bm25":
        return _bm25(engine, query, k=k)
    if mode == "hybrid":
        return _hybrid(engine, embedder, query, k=k)
    if mode == "graph":
        return _graph(engine, embedder, query, k=k, hops=hops)
    return _full(engine, embedder, query, k=k, hops=hops)


def _semantic(engine: Engine, embedder: Embedder, query: str, *, k: int) -> list[Fact]:
    emb = embedder.embed([query])[0]
    sql = text("""
        SELECT
            id, text, fact_type, source_chunk_id, temporal_start, temporal_end,
            confidence, tags, metadata, created_at,
            1 - (embedding <=> (:emb)::vector) AS score
        FROM facts
        ORDER BY embedding <=> (:emb)::vector
        LIMIT :k
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"emb": _emb_str(emb), "k": k}).fetchall()
    return [row_to_fact(r) for r in rows]


def _bm25(engine: Engine, query: str, *, k: int) -> list[Fact]:
    sql = text("""
        SELECT
            id, text, fact_type, source_chunk_id, temporal_start, temporal_end,
            confidence, tags, metadata, created_at,
            ts_rank_cd(tsv, plainto_tsquery('english', :query)) AS score
        FROM facts
        WHERE tsv @@ plainto_tsquery('english', :query)
        ORDER BY score DESC
        LIMIT :k
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"query": query, "k": k}).fetchall()
    return [row_to_fact(r) for r in rows]


def _hybrid(engine: Engine, embedder: Embedder, query: str, *, k: int) -> list[Fact]:
    candidate_k = k * 3
    sem = _semantic(engine, embedder, query, k=candidate_k)
    bm = _bm25(engine, query, k=candidate_k)

    sem_ids = [f.id for f in sem]
    bm_ids = [f.id for f in bm]
    scores = rrf([sem_ids, bm_ids])
    top_ids = sorted(scores, key=lambda d: scores[d], reverse=True)[:k]

    by_id: dict[UUID, Fact] = {f.id: f for f in sem}
    by_id.update({f.id: f for f in bm})

    return [
        replace(by_id[uid], score=scores[uid])
        for uid in top_ids
        if uid in by_id
    ]


def _graph(
    engine: Engine, embedder: Embedder, query: str, *, k: int, hops: int
) -> list[Fact]:
    candidate_k = k * 3
    seed_facts = _hybrid(engine, embedder, query, k=candidate_k)
    if not seed_facts:
        return seed_facts

    seed_fact_strs = [str(f.id) for f in seed_facts]
    seed_ids_pg = _ids_pg(seed_fact_strs)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT entity_id FROM fact_entities "
                "WHERE fact_id = ANY(CAST(:ids AS uuid[]))"
            ),
            {"ids": seed_ids_pg},
        ).fetchall()
    seed_entity_ids = [UUID(str(r.entity_id)) for r in rows]

    neighbor_entities = expand(engine, seed_entity_ids, hops=hops)
    neighbor_entity_ids = [e.id for e in neighbor_entities]
    neighbor_score_by_entity = {e.id: e.score for e in neighbor_entities}

    all_entity_ids = list({*seed_entity_ids, *neighbor_entity_ids})
    all_entity_strs = [str(eid) for eid in all_entity_ids]

    if not all_entity_strs:
        return seed_facts[:k]

    all_eids_pg = _ids_pg(all_entity_strs)

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT
                    f.id, f.text, f.fact_type, f.source_chunk_id,
                    f.temporal_start, f.temporal_end, f.confidence,
                    f.tags, f.metadata, f.created_at,
                    0.0 AS score
                FROM facts f
                JOIN fact_entities fe ON fe.fact_id = f.id
                WHERE fe.entity_id = ANY(CAST(:eids AS uuid[]))
            """),
            {"eids": all_eids_pg},
        ).fetchall()
    all_candidate_facts = [row_to_fact(r) for r in rows]

    entity_base_scores = {eid: 1.0 for eid in seed_entity_ids}
    entity_base_scores.update(neighbor_score_by_entity)

    scored_facts: dict[UUID, Fact] = {f.id: f for f in seed_facts}
    for fact in all_candidate_facts:
        if fact.id in scored_facts:
            continue
        with engine.connect() as conn:
            fe_rows = conn.execute(
                text("SELECT entity_id FROM fact_entities WHERE fact_id = :fid"),
                {"fid": str(fact.id)},
            ).fetchall()
        ent_ids = [UUID(str(r.entity_id)) for r in fe_rows]
        entity_score = max(
            (entity_base_scores[eid] for eid in ent_ids if eid in entity_base_scores),
            default=0.0,
        )
        scored_facts[fact.id] = replace(fact, score=entity_score)

    return sorted(scored_facts.values(), key=lambda f: f.score, reverse=True)[:k]


def _full(
    engine: Engine, embedder: Embedder, query: str, *, k: int, hops: int
) -> list[Fact]:
    from .rerank import rerank

    candidate_k = max(k * 5, 50)
    graph_results = _graph(engine, embedder, query, k=candidate_k, hops=hops)
    if not graph_results:
        return []
    return rerank(query, graph_results, k=k)
