"""
Integration tests — ingestion pipeline (parse → chunk → embed → index).

Tests the full pipeline sequence end-to-end with all Azure calls mocked.
Validates that:
  - Data flows correctly from parse_pdf through to upload_chunks
  - chunk_ids are correctly constructed and unique
  - All chunks carry the doc_id from the parent document
  - Embedding dimension is preserved through the pipeline
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


FAKE_PDF_BYTES = b"%PDF-1.4 fake content"
FAKE_DOC_ID = "integration-doc-001"
FAKE_FILENAME = "clinical_protocol.pdf"


@pytest.fixture
def mock_fitz_doc():
    """Fake fitz document with 2 pages of text."""
    page1 = MagicMock()
    page1.get_text.return_value = (
        "The study enrolled 500 patients across 12 sites. "
        "Primary endpoint: HbA1c reduction at 24 weeks. " * 20
    )
    page1.get_images.return_value = []

    page2 = MagicMock()
    page2.get_text.return_value = (
        "Secondary endpoints included body weight and fasting glucose. "
        "Statistical analysis used mixed-effects model. " * 20
    )
    page2.get_images.return_value = []

    doc = MagicMock()
    doc.__iter__ = MagicMock(return_value=iter([page1, page2]))
    doc.close = MagicMock()
    return doc


class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_produces_chunks_with_embeddings(self, mock_fitz_doc):
        """parse → chunk → embed → verifies output structure."""
        # 1. Parse
        with patch("fitz.open", return_value=mock_fitz_doc):
            from ingestion.parser import parse_pdf
            parsed = parse_pdf(FAKE_PDF_BYTES)

        assert parsed["total_pages"] == 2
        assert all(p["text"] for p in parsed["pages"])

        # 2. Chunk
        from ingestion.chunker import chunk_text
        chunks = chunk_text(parsed["pages"], FAKE_DOC_ID, FAKE_FILENAME)

        assert len(chunks) > 0
        assert all(c["doc_id"] == FAKE_DOC_ID for c in chunks)
        assert all(c["filename"] == FAKE_FILENAME for c in chunks)

        # chunk_ids must be unique
        chunk_ids = [c["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

        # 3. Embed (mocked)
        def fake_embed_response(n):
            r = MagicMock()
            r.data = [MagicMock() for _ in range(n)]
            for obj in r.data:
                obj.embedding = [0.01] * 1536
            return r

        mock_oai_client = MagicMock()
        mock_oai_client.embeddings.create = AsyncMock(
            side_effect=lambda **kw: fake_embed_response(len(kw["input"]))
        )

        with patch("openai.AsyncAzureOpenAI", return_value=mock_oai_client):
            from ingestion.embedder import embed_chunks
            embedded = await embed_chunks(chunks)

        assert all("embedding" in c for c in embedded)
        assert all(len(c["embedding"]) == 1536 for c in embedded)

    @pytest.mark.asyncio
    async def test_chunk_ids_span_both_pages(self, mock_fitz_doc):
        """Chunks should come from both page 1 and page 2."""
        with patch("fitz.open", return_value=mock_fitz_doc):
            from ingestion.parser import parse_pdf
            parsed = parse_pdf(FAKE_PDF_BYTES)

        from ingestion.chunker import chunk_text
        chunks = chunk_text(parsed["pages"], FAKE_DOC_ID, FAKE_FILENAME)

        page_numbers = {c["page_number"] for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    @pytest.mark.asyncio
    async def test_indexer_upload_called_with_correct_fields(self, mock_fitz_doc):
        """upload_chunks should receive all required fields for Azure AI Search."""
        with patch("fitz.open", return_value=mock_fitz_doc):
            from ingestion.parser import parse_pdf
            parsed = parse_pdf(FAKE_PDF_BYTES)

        from ingestion.chunker import chunk_text
        chunks = chunk_text(parsed["pages"], FAKE_DOC_ID, FAKE_FILENAME)

        # Add fake embeddings
        for c in chunks:
            c["embedding"] = [0.0] * 1536

        uploaded_docs = []

        mock_search_client = MagicMock()
        mock_search_client.upload_documents = MagicMock(
            side_effect=lambda documents: uploaded_docs.extend(documents)
        )

        with patch("ingestion.indexer.get_search_client", return_value=mock_search_client):
            from ingestion.indexer import upload_chunks
            await upload_chunks(chunks)

        assert len(uploaded_docs) == len(chunks)
        required_fields = {"chunk_id", "doc_id", "filename", "page_number", "text", "embedding"}
        for doc in uploaded_docs:
            assert required_fields.issubset(set(doc.keys()))
