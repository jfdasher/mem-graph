from __future__ import annotations

from pathlib import Path

from . import pdf, text

_PARSERS = {
    ".md": text.parse,
    ".txt": text.parse,
    ".pdf": pdf.parse,
}


def parse(path: str | Path) -> str:
    """Dispatch to the appropriate parser based on file extension.

    Raises ValueError for unsupported extensions.
    """
    p = Path(path)
    parser = _PARSERS.get(p.suffix.lower())
    if parser is None:
        raise ValueError(f"Unsupported file extension: {p.suffix!r} (file: {p})")
    return parser(p)
