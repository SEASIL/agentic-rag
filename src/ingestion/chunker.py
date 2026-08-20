"""
Splits raw text records (from pdf_loader) into overlapping chunks sized for
the embedding model's context window. Table/API records are already
chunk-sized by their loaders and pass through untouched.
"""
from __future__ import annotations

from src.ingestion.schema import Chunk, SourceType
from configs.settings import settings


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window word-based splitter. Swap for a token-aware
    splitter (e.g. tiktoken-based) if precise context-window budgeting matters."""
    words = text.split()
    if not words:
        return []

    chunks, start = [], 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        start += step
    return chunks


def records_to_chunks(records: list[dict]) -> list[Chunk]:
    """Turns raw loader output into final Chunk objects, applying text
    splitting only to prose (PDF/web) records — tables/API JSON pass through."""
    chunks: list[Chunk] = []

    for rec in records:
        source_type = rec["source_type"]

        if source_type in (SourceType.TABLE, SourceType.API_JSON):
            locator = f"{rec.get('metadata', {}).get('part_index', 0)}"
            chunks.append(
                Chunk(
                    id=Chunk.make_id(rec["source_path"], locator),
                    text=rec["text"],
                    source_type=source_type,
                    source_path=rec["source_path"],
                    page_number=rec.get("page_number"),
                    metadata=rec.get("metadata", {}),
                )
            )
            continue

        # Prose: apply sliding-window split
        pieces = _split_text(rec["text"], settings.chunk_size, settings.chunk_overlap)
        for i, piece in enumerate(pieces):
            locator = f"p{rec.get('page_number')}_c{i}"
            chunks.append(
                Chunk(
                    id=Chunk.make_id(rec["source_path"], locator),
                    text=piece,
                    source_type=source_type,
                    source_path=rec["source_path"],
                    page_number=rec.get("page_number"),
                    metadata={"chunk_index": i},
                )
            )
    return chunks
