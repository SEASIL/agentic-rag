"""
Table ingestion. Tables break naive text chunking, so each table becomes ONE
chunk (or a few, if very large) using one of three strategies configured in
settings.table_chunk_strategy:

  - structured_json: keep rows as JSON in metadata, text = markdown preview
                      (best when a downstream agent will programmatically filter rows)
  - markdown:         full table rendered as markdown text (best for small tables,
                      lets the LLM read it directly, and it's dense-embedding friendly)
  - summarize:        an LLM-generated natural-language summary of the table
                      (best for very large tables where raw rows would blow the
                      context window; requires an LLM call, stubbed here)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.ingestion.schema import SourceType
from configs.settings import settings

MAX_ROWS_PER_CHUNK = 200  # split very large tables across multiple chunks


def load_table(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported table format: {file_path.suffix}")


def _row_chunks(df: pd.DataFrame, max_rows: int) -> list[pd.DataFrame]:
    return [df.iloc[i : i + max_rows] for i in range(0, len(df), max_rows)]


def table_to_raw_records(file_path: str | Path, strategy: str | None = None) -> list[dict]:
    file_path = Path(file_path)
    strategy = strategy or settings.table_chunk_strategy
    df = load_table(file_path)

    records = []
    for part_idx, part in enumerate(_row_chunks(df, MAX_ROWS_PER_CHUNK)):
        if strategy == "markdown":
            text = part.to_markdown(index=False)
            metadata = {"columns": list(df.columns), "row_range": [part.index.min(), part.index.max()]}

        elif strategy == "structured_json":
            # Text kept human/LLM-readable; full fidelity data lives in metadata
            # so an agent could filter/aggregate rows programmatically later.
            text = f"Table with columns: {', '.join(df.columns)}\n" + part.head(10).to_markdown(index=False)
            metadata = {
                "columns": list(df.columns),
                "row_range": [int(part.index.min()), int(part.index.max())],
                "rows": part.to_dict(orient="records"),
            }

        elif strategy == "summarize":
            # Placeholder — wire this to an LLM call in production.
            text = f"[SUMMARY PLACEHOLDER] Table with {len(df)} rows and columns: {', '.join(df.columns)}"
            metadata = {"columns": list(df.columns), "row_range": [int(part.index.min()), int(part.index.max())]}

        else:
            raise ValueError(f"Unknown table_chunk_strategy: {strategy}")

        records.append(
            {
                "text": text,
                "source_type": SourceType.TABLE,
                "source_path": str(file_path),
                "page_number": None,
                "metadata": {**metadata, "part_index": part_idx},
            }
        )
    return records
