from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import Engine
from sqlalchemy import text as _text

if TYPE_CHECKING:
    from .embedders import Embedder
    from .llm import LLMProvider


class ExtractedEntity(BaseModel):
    name: str
    type: Literal["person", "place", "org", "concept", "artifact", "event"]


class ExtractedFact(BaseModel):
    text: str
    fact_type: Literal["world", "experience"]
    entities: list[ExtractedEntity]
    temporal_start: datetime | None = None
    temporal_end: datetime | None = None
    confidence: float = Field(ge=0, le=1, default=0.8)


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact]


def parse_extraction(raw: dict[str, Any]) -> ExtractionResult:
    try:
        return ExtractionResult.model_validate(raw)
    except ValidationError:
        return ExtractionResult(facts=[])
    except Exception:
        return ExtractionResult(facts=[])


def _emb_str(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


def extract_facts_with_entities(
    engine: Engine,
    embedder: Embedder,
    llm: LLMProvider,
    source_chunk_id: UUID,
    chunk_text: str,
) -> list[tuple[UUID, list[ExtractedEntity]]]:
    """Extract and write facts; return (fact_id, entities) tuples for caller to resolve."""
    raw = llm.extract_facts(chunk_text)
    result = parse_extraction(raw)

    if not result.facts:
        return []

    fact_texts = [f.text for f in result.facts]
    embeddings = embedder.embed(fact_texts)

    records: list[tuple[UUID, list[ExtractedEntity]]] = []
    with engine.connect() as conn:
        for fact, embedding in zip(result.facts, embeddings, strict=True):
            row = conn.execute(
                _text("""
                    INSERT INTO facts
                        (text, fact_type, source_chunk_id, temporal_start,
                         temporal_end, confidence, embedding)
                    VALUES
                        (:text, :fact_type, :source_chunk_id, :temporal_start,
                         :temporal_end, :confidence, (:embedding)::vector)
                    RETURNING id
                """),
                {
                    "text": fact.text,
                    "fact_type": fact.fact_type,
                    "source_chunk_id": str(source_chunk_id),
                    "temporal_start": fact.temporal_start,
                    "temporal_end": fact.temporal_end,
                    "confidence": fact.confidence,
                    "embedding": _emb_str(embedding),
                },
            ).fetchone()
            assert row is not None
            records.append((UUID(str(row[0])), fact.entities))
        conn.commit()

    return records


def extract_facts_from_chunk(
    engine: Engine,
    embedder: Embedder,
    llm: LLMProvider,
    source_chunk_id: UUID,
    chunk_text: str,
) -> list[UUID]:
    """Extract facts via LLM, embed each, and write to facts table.

    Does NOT resolve entities or write fact_entities — done by ingest_chunk (Task 5).
    Returns list of inserted fact UUIDs.
    """
    raw = llm.extract_facts(chunk_text)
    result = parse_extraction(raw)

    if not result.facts:
        return []

    fact_texts = [f.text for f in result.facts]
    embeddings = embedder.embed(fact_texts)

    fact_ids: list[UUID] = []
    with engine.connect() as conn:
        for fact, embedding in zip(result.facts, embeddings, strict=True):
            row = conn.execute(
                _text("""
                    INSERT INTO facts
                        (text, fact_type, source_chunk_id, temporal_start,
                         temporal_end, confidence, embedding)
                    VALUES
                        (:text, :fact_type, :source_chunk_id, :temporal_start,
                         :temporal_end, :confidence, (:embedding)::vector)
                    RETURNING id
                """),
                {
                    "text": fact.text,
                    "fact_type": fact.fact_type,
                    "source_chunk_id": str(source_chunk_id),
                    "temporal_start": fact.temporal_start,
                    "temporal_end": fact.temporal_end,
                    "confidence": fact.confidence,
                    "embedding": _emb_str(embedding),
                },
            ).fetchone()
            assert row is not None
            fact_ids.append(UUID(str(row[0])))
        conn.commit()

    return fact_ids
