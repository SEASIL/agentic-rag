from src.ingestion.chunker import records_to_chunks, _split_text
from src.ingestion.schema import SourceType


def test_split_text_respects_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    pieces = _split_text(text, chunk_size=20, overlap=5)

    # step = 15, so piece[1] should start 15 words into piece[0]
    first_words = pieces[0].split()
    second_words = pieces[1].split()
    assert first_words[15:20] == second_words[0:5]


def test_records_to_chunks_prose_gets_split():
    long_text = " ".join(f"word{i}" for i in range(2000))
    records = [
        {
            "text": long_text,
            "source_type": SourceType.PDF_TEXT,
            "source_path": "doc.pdf",
            "page_number": 1,
        }
    ]
    chunks = records_to_chunks(records)
    assert len(chunks) > 1
    assert all(c.source_type == SourceType.PDF_TEXT for c in chunks)


def test_records_to_chunks_table_passes_through_unsplit():
    records = [
        {
            "text": "table preview",
            "source_type": SourceType.TABLE,
            "source_path": "data.csv",
            "page_number": None,
            "metadata": {"part_index": 0},
        }
    ]
    chunks = records_to_chunks(records)
    assert len(chunks) == 1
    assert chunks[0].text == "table preview"


def test_chunk_ids_are_deterministic():
    records = [
        {
            "text": "hello world " * 10,
            "source_type": SourceType.PDF_TEXT,
            "source_path": "doc.pdf",
            "page_number": 1,
        }
    ]
    chunks_a = records_to_chunks(records)
    chunks_b = records_to_chunks(records)
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
