from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

EXTRACTION_SYSTEM_PROMPT = """\
Extract atomic facts from the text. For each fact:
- text: one concise sentence
- fact_type: "world" (general knowledge) or "experience" (first-person, dated event)
- entities: list of named entities mentioned, each with name and type
  (type: person | place | org | concept | artifact | event)
- temporal_start / temporal_end: ISO datetime strings if applicable, else null
- confidence: 0.0-1.0

Return ONLY valid JSON: {"facts": [...]}
"""


class LLMProvider(ABC):
    @abstractmethod
    def extract_facts(self, chunk_text: str) -> dict[str, Any]:
        """Return raw dict matching ExtractionResult shape."""


class MockLLM(LLMProvider):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._queue = list(responses)

    def extract_facts(self, chunk_text: str) -> dict[str, Any]:
        if not self._queue:
            return {"facts": []}
        return self._queue.pop(0)


class OpenAILLM(LLMProvider):
    _TOOL_SCHEMA: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "record_facts",
            "description": "Record extracted facts from the text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "fact_type": {
                                    "type": "string",
                                    "enum": ["world", "experience"],
                                },
                                "entities": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {
                                                "type": "string",
                                                "enum": [
                                                    "person",
                                                    "place",
                                                    "org",
                                                    "concept",
                                                    "artifact",
                                                    "event",
                                                ],
                                            },
                                        },
                                        "required": ["name", "type"],
                                    },
                                },
                                "temporal_start": {"type": ["string", "null"]},
                                "temporal_end": {"type": ["string", "null"]},
                                "confidence": {"type": "number"},
                            },
                            "required": ["text", "fact_type", "entities"],
                        },
                    }
                },
                "required": ["facts"],
            },
        },
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._extra_headers = extra_headers or {}
        self._client: Any = None

    def _init(self) -> None:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                default_headers=self._extra_headers,
            )

    def extract_facts(self, chunk_text: str) -> dict[str, Any]:
        self._init()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": chunk_text},
            ],
            tools=[self._TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "record_facts"}},
        )
        message = response.choices[0].message
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            try:
                return json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                pass
        return {"facts": []}
