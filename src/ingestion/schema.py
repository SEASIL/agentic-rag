"""Shared data model for ingested chunks, regardless of source type."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import hashlib


class SourceType(str, Enum):
    PDF_TEXT = "pdf_text"
    TABLE = "table"
    API_JSON = "api_json"
    WEB = "web"


class Chunk(BaseModel):
    """A single retrievable unit. Tables and text share this shape so the
    retriever never needs to know what produced a chunk."""

    id: str
    text: str                              # the string that gets embedded
    source_type: SourceType
    source_path: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # e.g. table schema, row range

    @staticmethod
    def make_id(source_path: str, locator: str) -> str:
        """Deterministic id so re-ingesting the same doc doesn't duplicate points."""
        raw = f"{source_path}::{locator}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
