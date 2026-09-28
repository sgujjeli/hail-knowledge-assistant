"""
Unit tests — ingestion/chunker.py

Tests:
  - Chunks are created for normal text
  - chunk_id format is correct ({doc_id}_p{page}_c{index})
  - Blank pages are skipped
  - Overlap is respected (text from end of chunk N appears in chunk N+1)
  - All chunks reference the correct doc_id and filename
  - Very short text produces exactly 1 chunk
"""
import pytest
from ingestion.chunker import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP


def _make_pages(texts: list[str]) -> list[dict]:
    return [
        {"page_number": i + 1, "text": t, "images": [], "char_count": len(t)}
        for i, t in enumerate(texts)
    ]


class TestChunkText:
    def test_returns_chunks_for_normal_text(self):
        pages = _make_pages(["The quick brown fox jumps over the lazy dog. " * 30])
        chunks = chunk_text(pages, "docA", "test.pdf")
        assert len(chunks) >= 1

    def test_chunk_id_format(self):
        pages = _make_pages(["Hello world. " * 10])
        chunks = chunk_text(pages, "docA", "test.pdf")
        for chunk in chunks:
            parts = chunk["chunk_id"].split("_")
            assert parts[0] == "docA"
            assert parts[1].startswith("p")
            assert parts[2].startswith("c")

    def test_blank_pages_skipped(self):
        pages = _make_pages(["   ", "\n\n", "Actual content here."])
        chunks = chunk_text(pages, "docB", "blank.pdf")
        # Only page 3 has content
        assert all(c["page_number"] == 3 for c in chunks)
        assert len(chunks) == 1

    def test_doc_id_and_filename_propagated(self):
        pages = _make_pages(["Some text. " * 20])
        chunks = chunk_text(pages, "my-doc-id", "report.pdf")
        for chunk in chunks:
            assert chunk["doc_id"] == "my-doc-id"
            assert chunk["filename"] == "report.pdf"

    def test_short_text_produces_one_chunk(self):
        pages = _make_pages(["This is a very short paragraph."])
        chunks = chunk_text(pages, "docC", "short.pdf")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "This is a very short paragraph."

    def test_long_text_produces_multiple_chunks(self):
        # 5× CHUNK_SIZE should create at least 4 chunks
        long_text = "word " * (CHUNK_SIZE * 5 // 5)
        pages = _make_pages([long_text])
        chunks = chunk_text(pages, "docD", "long.pdf")
        assert len(chunks) >= 4

    def test_overlap_present_between_adjacent_chunks(self):
        """Last tokens of chunk N should appear in chunk N+1."""
        # Create text long enough for 2 chunks
        long_text = ("The primary endpoint was a reduction in adverse events. " * 30)
        pages = _make_pages([long_text])
        chunks = chunk_text(pages, "docE", "overlap.pdf")
        if len(chunks) < 2:
            pytest.skip("Text too short for overlap test")
        tail_of_first = chunks[0]["text"][-CHUNK_OVERLAP:]
        assert any(tail_of_first[:20] in chunks[1]["text"] for _ in [1])

    def test_char_count_matches_text_length(self):
        pages = _make_pages(["Hello there. " * 10])
        chunks = chunk_text(pages, "docF", "count.pdf")
        for chunk in chunks:
            assert chunk["char_count"] == len(chunk["text"])

    def test_multi_page_document(self):
        pages = _make_pages(["Page one content. " * 10, "Page two content. " * 10])
        chunks = chunk_text(pages, "docG", "multi.pdf")
        page_numbers = {c["page_number"] for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers
