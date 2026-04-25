from __future__ import annotations

from pathlib import Path


def parse(path: Path) -> str:
    """Read a plain-text or Markdown file, tolerating encoding errors."""
    return path.read_text(encoding="utf-8", errors="replace")
