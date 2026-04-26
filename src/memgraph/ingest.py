from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Engine, text

from .chunker import word_chunks
from .embedders import Embedder
from .extract import extract_facts_with_entities
from .graph import upsert_entity_links
from .parsers import parse as parse_file
from .resolve import resolve_or_create

if TYPE_CHECKING:
    from .llm import LLMProvider

logger = logging.getLogger(__name__)


def _emb_str(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


def ingest_chunk(
    engine: Engine,
    embedder: Embedder,
    llm: LLMProvider,
    content: str,
    *,
    document_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> UUID:
    """Embed content, insert chunk, extract facts, resolve entities, upsert links."""
    logger.debug("ingest_chunk: doc_id=%r, %d chars", document_id, len(content))
    embedding = embedder.embed([content])[0]
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "INSERT INTO chunks (text, embedding, document_id, tags, metadata) "
                "VALUES (:text, (:embedding)::vector, :document_id, :tags, :metadata) "
                "RETURNING id"
            ),
            {
                "text": content,
                "embedding": _emb_str(embedding),
                "document_id": document_id,
                "tags": tags or [],
                "metadata": json.dumps(metadata or {}),
            },
        ).fetchone()
        conn.commit()
    assert row is not None
    chunk_id = UUID(str(row[0]))

    fact_records = extract_facts_with_entities(engine, embedder, llm, chunk_id, content)
    logger.debug("ingest_chunk: chunk %s -> %d facts", chunk_id, len(fact_records))

    for fact_id, extracted_entities in fact_records:
        entity_ids: list[UUID] = []
        for ent in extracted_entities:
            eid = resolve_or_create(engine, ent.name, ent.type)
            with engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO fact_entities (fact_id, entity_id)
                        VALUES (:fact_id, :entity_id)
                        ON CONFLICT DO NOTHING
                    """),
                    {"fact_id": str(fact_id), "entity_id": str(eid)},
                )
                conn.commit()
            entity_ids.append(eid)

        upsert_entity_links(engine, entity_ids)

    return chunk_id


def ingest_file(
    engine: Engine,
    embedder: Embedder,
    llm: LLMProvider,
    path: str | Path,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    recursive: bool = True,
    tags: list[str] | None = None,
) -> list[UUID]:
    """Walk path, parse, chunk, and ingest. Returns list of chunk UUIDs."""
    root = Path(path).resolve()
    if root.is_file():
        files = [root]
    else:
        glob = "**/*" if recursive else "*"
        files = sorted(f for f in root.glob(glob) if f.is_file())

    ids: list[UUID] = []
    for file_path in files:
        logger.debug("ingest_file: %s", file_path)
        try:
            text_content = parse_file(file_path)
        except Exception:
            continue
        chunks = word_chunks(text_content, chunk_size=chunk_size, overlap=overlap)
        doc_id = str(file_path.relative_to(root)) if root.is_dir() else str(file_path.name)
        meta: dict[str, object] = {"source_path": str(file_path)}
        for chunk in chunks:
            uid = ingest_chunk(
                engine, embedder, llm, chunk,
                document_id=doc_id, tags=tags, metadata=meta,
            )
            ids.append(uid)
    return ids
