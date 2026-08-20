"""
PDF ingestion: extracts page-level text, preserving page numbers for citation.
Tables embedded in PDFs are handled separately by table_loader.py when possible
(via `unstructured`'s partitioning); this module focuses on prose text.
"""
from __future__ import annotations

from pathlib import Path
from pypdf import PdfReader

from src.ingestion.schema import Chunk, SourceType


def load_pdf_pages(file_path: str | Path) -> list[dict]:
    """Returns raw page records: [{page_number, text}, ...].
    Kept separate from chunking so callers can apply custom chunking strategies."""
    file_path = Path(file_path)
    reader = PdfReader(str(file_path))

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_number": i, "text": text})
    return pages


def pdf_to_raw_records(file_path: str | Path) -> list[dict]:
    """Convenience wrapper -> list of dicts ready for the chunker."""
    file_path = Path(file_path)
    pages = load_pdf_pages(file_path)
    return [
        {
            "text": p["text"],
            "source_type": SourceType.PDF_TEXT,
            "source_path": str(file_path),
            "page_number": p["page_number"],
        }
        for p in pages
    ]
