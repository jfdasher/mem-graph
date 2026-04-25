from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, text


@dataclass(frozen=True)
class Chunk:
    id: UUID
    text: str
    document_id: str | None
    tags: list[str]
    metadata: dict[str, Any]
    ingested_at: datetime
    score: float = 0.0


@dataclass(frozen=True)
class Fact:
    id: UUID
    text: str
    fact_type: str
    source_chunk_id: UUID
    temporal_start: datetime | None
    temporal_end: datetime | None
    confidence: float | None
    tags: list[str]
    metadata: dict[str, Any]
    created_at: datetime
    score: float = 0.0


@dataclass(frozen=True)
class Entity:
    id: UUID
    name: str
    type: str | None
    aliases: list[str]
    created_at: datetime


@dataclass
class Filters:
    document_id: str | None = None
    ingested_at_from: datetime | None = None
    ingested_at_to: datetime | None = None
    tags: list[str] | None = None


def create_schema(engine: Engine, dimension: int) -> None:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    text        TEXT NOT NULL,
                    embedding   VECTOR({dimension}) NOT NULL,
                    tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
                    document_id TEXT,
                    tags        TEXT[] NOT NULL DEFAULT '{{}}',
                    metadata    JSONB NOT NULL DEFAULT '{{}}',
                    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_ingested_at ON chunks(ingested_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks(document_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_tags ON chunks USING GIN(tags)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING GIN(tsv)"))
        conn.commit()


def create_graph_schema(engine: Engine, dimension: int) -> None:
    pass  # implemented in Task 1


def row_to_chunk(row: Any) -> Chunk:
    meta = row.metadata
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Chunk(
        id=UUID(str(row.id)),
        text=row.text,
        document_id=row.document_id,
        tags=list(row.tags) if row.tags else [],
        metadata=meta,
        ingested_at=row.ingested_at,
        score=float(row.score),
    )


def row_to_fact(row: Any) -> Fact:
    meta = row.metadata
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Fact(
        id=UUID(str(row.id)),
        text=row.text,
        fact_type=row.fact_type,
        source_chunk_id=UUID(str(row.source_chunk_id)),
        temporal_start=row.temporal_start,
        temporal_end=row.temporal_end,
        confidence=float(row.confidence) if row.confidence is not None else None,
        tags=list(row.tags) if row.tags else [],
        metadata=meta,
        created_at=row.created_at,
        score=float(getattr(row, "score", 0.0)),
    )


def row_to_entity(row: Any) -> Entity:
    return Entity(
        id=UUID(str(row.id)),
        name=row.name,
        type=row.type,
        aliases=list(row.aliases) if row.aliases else [],
        created_at=row.created_at,
    )
