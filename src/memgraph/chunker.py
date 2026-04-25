from __future__ import annotations


def word_chunks(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split *text* into overlapping word-count windows.

    Returns an empty list if *text* is empty or whitespace-only.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks
