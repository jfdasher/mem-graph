from __future__ import annotations

import logging
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)


class EncryptedPDFError(ValueError):
    """Raised when a PDF is password-protected."""


def parse(path: Path) -> str:
    """Extract text from a PDF file.

    Returns an empty string (with a warning) for image-only PDFs.
    Raises EncryptedPDFError for password-protected PDFs.
    """
    try:
        reader = pypdf.PdfReader(path)
    except pypdf.errors.FileNotDecryptedError as exc:
        raise EncryptedPDFError(f"PDF is encrypted: {path}") from exc

    # is_encrypted is set during construction; check it before accessing page objects
    if reader.is_encrypted:
        raise EncryptedPDFError(f"PDF is encrypted: {path}")

    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        logger.warning("No text extracted from %s (image-only or empty PDF)", path)
    return text
