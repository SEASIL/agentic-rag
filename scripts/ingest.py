"""
Ingest all documents in data/raw/ into Qdrant.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --path data/raw/some_folder
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to python path so 'src' can be imported
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.pdf_loader import pdf_to_raw_records
from src.ingestion.table_loader import table_to_raw_records
from src.ingestion.chunker import records_to_chunks
from src.retrieval.vector_store import upsert_chunks, ensure_collection

LOADERS = {
    ".pdf": pdf_to_raw_records,
    ".csv": table_to_raw_records,
    ".xlsx": table_to_raw_records,
    ".xls": table_to_raw_records,
}


def ingest_path(root: Path) -> int:
    ensure_collection()

    total_chunks = 0
    files = [f for f in root.rglob("*") if f.suffix.lower() in LOADERS]

    if not files:
        print(f"No supported files found under {root} (looked for {list(LOADERS)})")
        return 0

    for file_path in files:
        loader = LOADERS[file_path.suffix.lower()]
        print(f"Ingesting {file_path} ...")
        records = loader(file_path)
        chunks = records_to_chunks(records)
        upsert_chunks(chunks)
        total_chunks += len(chunks)
        print(f"  -> {len(chunks)} chunks")

    return total_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/raw", help="Folder to ingest recursively")
    args = parser.parse_args()

    n = ingest_path(Path(args.path))
    print(f"\nDone. Ingested {n} total chunks.")
