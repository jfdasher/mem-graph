from __future__ import annotations

from typing import Any

SIMPLE_WORLD: dict[str, Any] = {
    "facts": [
        {
            "text": "The sky is blue.",
            "fact_type": "world",
            "entities": [{"name": "sky", "type": "concept"}],
            "temporal_start": None,
            "temporal_end": None,
            "confidence": 0.9,
        }
    ]
}

EXPERIENCE_FACT: dict[str, Any] = {
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

MULTI_ENTITY: dict[str, Any] = {
    "facts": [
        {
            "text": "Alice works with Bob on the Orion project.",
            "fact_type": "experience",
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
                {"name": "Orion project", "type": "artifact"},
            ],
            "temporal_start": None,
            "temporal_end": None,
            "confidence": 0.88,
        }
    ]
}

EMPTY: dict[str, Any] = {"facts": []}

MALFORMED: dict[str, Any] = {"wrong_key": "bad value"}

ALICE_BOB_COLAB: dict[str, Any] = {
    "facts": [
        {
            "text": "Alice collaborated with Bob on the merger analysis.",
            "fact_type": "experience",
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
            ],
            "temporal_start": None,
            "temporal_end": None,
            "confidence": 0.9,
        }
    ]
}

BOB_DEADLINE: dict[str, Any] = {
    "facts": [
        {
            "text": "Bob's merger analysis deadline is next Friday.",
            "fact_type": "experience",
            "entities": [
                {"name": "Bob", "type": "person"},
            ],
            "temporal_start": None,
            "temporal_end": None,
            "confidence": 0.85,
        }
    ]
}
