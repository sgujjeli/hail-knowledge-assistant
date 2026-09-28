"""
Unit tests — ingestion/embedder.py

Tests:
  - embed_chunks: calls the Azure OpenAI endpoint with batches of 100
  - embed_chunks: each chunk receives an 'embedding' key of 1536 floats
  - embed_chunks: batching works correctly for > 100 chunks
  - embed_single_text: returns a flat list of 1536 floats
  - endpoint URL contains the correct deployment name
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_fake_embedding_response(n_texts: int):
    """Return a fake OpenAI embeddings response for n_texts inputs."""
    response = MagicMock()
    response.data = [MagicMock() for _ in range(n_texts)]
    for i, obj in enumerate(response.data):
        obj.embedding = [float(i) / 1536] * 1536
    return response


class TestEmbedChunks:
    @pytest.mark.asyncio
    async def test_adds_embedding_to_each_chunk(self):
        chunks = [
            {"chunk_id": f"doc_p1_c{i}", "text": f"text {i}"}
            for i in range(3)
        ]
        fake_response = _make_fake_embedding_response(3)

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=fake_response)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from ingestion.embedder import embed_chunks
            result = await embed_chunks(chunks)

        for chunk in result:
            assert "embedding" in chunk
            assert len(chunk["embedding"]) == 1536

    @pytest.mark.asyncio
    async def test_batches_correctly_for_over_100_chunks(self):
        """130 chunks should produce 2 API calls (100 + 30)."""
        chunks = [{"chunk_id": f"doc_p1_c{i}", "text": f"text {i}"} for i in range(130)]
        call_count = 0

        async def fake_create(**kwargs):
            nonlocal call_count
            n = len(kwargs["input"])
            call_count += 1
            return _make_fake_embedding_response(n)

        mock_client = MagicMock()
        mock_client.embeddings.create = fake_create

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from ingestion.embedder import embed_chunks
            await embed_chunks(chunks)

        assert call_count == 2  # batch 1: 100, batch 2: 30

    @pytest.mark.asyncio
    async def test_embedding_dimension_is_1536(self):
        chunks = [{"chunk_id": "doc_p1_c0", "text": "test text"}]
        fake_response = _make_fake_embedding_response(1)

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=fake_response)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from ingestion.embedder import embed_chunks
            result = await embed_chunks(chunks)

        assert len(result[0]["embedding"]) == 1536

    @pytest.mark.asyncio
    async def test_embed_single_text_returns_list_of_floats(self):
        fake_response = _make_fake_embedding_response(1)

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=fake_response)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from ingestion.embedder import embed_single_text
            result = await embed_single_text("What is the primary endpoint?")

        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_correct_deployment_used(self):
        """embed_chunks should pass the configured deployment name, not a hardcoded string."""
        chunks = [{"chunk_id": "doc_p1_c0", "text": "hello"}]
        fake_response = _make_fake_embedding_response(1)

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=fake_response)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from ingestion.embedder import embed_chunks
            await embed_chunks(chunks)

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-small"
