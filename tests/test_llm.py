from __future__ import annotations

from memgraph.extract import parse_extraction
from memgraph.llm import MockLLM
from tests.fixtures.canned_llm import EMPTY, EXPERIENCE_FACT, MALFORMED, SIMPLE_WORLD


def test_mock_llm_returns_queued_responses() -> None:
    llm = MockLLM([SIMPLE_WORLD, EXPERIENCE_FACT])
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
