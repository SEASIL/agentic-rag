import pandas as pd
import pytest

from src.ingestion.table_loader import table_to_raw_records, _row_chunks


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({"id": range(5), "value": [f"v{i}" for i in range(5)]})
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_structured_json_strategy_preserves_all_rows(sample_csv):
    records = table_to_raw_records(sample_csv, strategy="structured_json")
    assert len(records) == 1
    assert len(records[0]["metadata"]["rows"]) == 5


def test_markdown_strategy_produces_readable_text(sample_csv):
    records = table_to_raw_records(sample_csv, strategy="markdown")
    assert "id" in records[0]["text"]
    assert "value" in records[0]["text"]


def test_row_chunks_splits_large_tables():
    df = pd.DataFrame({"x": range(450)})
    chunks = _row_chunks(df, max_rows=200)
    assert len(chunks) == 3
    assert len(chunks[0]) == 200
    assert len(chunks[-1]) == 50


def test_unknown_strategy_raises(sample_csv):
    with pytest.raises(ValueError):
        table_to_raw_records(sample_csv, strategy="not_a_real_strategy")
