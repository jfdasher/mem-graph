from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
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
    except (ValidationError, Exception):
        return ExtractionResult(facts=[])


# extract_facts_from_chunk() implemented in Task 3
