from __future__ import annotations

from tests.fixtures.canned_llm import EMPTY, MALFORMED, MULTI_ENTITY, SIMPLE_WORLD
from memgraph.llm import MockLLM
from memgraph.extract import ExtractionResult, parse_extraction


def test_mock_llm_returns_queued_responses() -> None:
    llm = MockLLM([SIMPLE_WORLD, EXPERIENCE_FACT_DATA])
    r1 = llm.extract_facts("chunk 1")
    r2 = llm.extract_facts("chunk 2")
    r3 = llm.extract_facts("chunk 3")  # queue exhausted → empty
    assert r1["facts"][0]["text"] == "The sky is blue."
    assert r2["facts"][0]["fact_type"] == "experience"
    assert r3 == {"facts": []}


def test_mock_llm_empty_response() -> None:
    llm = MockLLM([EMPTY])
    result = parse_extraction(llm.extract_facts("anything"))
    assert result.facts == []


def test_parse_extraction_handles_malformed() -> None:
    result = parse_extraction(MALFORMED)
    assert result.facts == []


EXPERIENCE_FACT_DATA = {
    "facts": [
        {
            "text": "Alice got promoted on 2025-06-15.",
            "fact_type": "experience",
            "entities": [{"name": "Alice", "type": "person"}],
            "temporal_start": "2025-06-15T00:00:00",
            "temporal_end": None,
            "confidence": 0.95,
        }
    ]
}
