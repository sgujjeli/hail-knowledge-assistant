"""
Unit tests — retrieval/hybrid_search.py

Tests:
  - hybrid_search: embeds the query and calls SearchClient.search
  - hybrid_search: passes the doc_id filter when provided
  - hybrid_search: returns correctly structured chunk dicts
  - hybrid_search: uses semantic reranker score when available
  - retrieval_quality_gate: returns True when score >= threshold
  - retrieval_quality_gate: returns False when score < threshold
  - retrieval_quality_gate: returns False for empty chunks
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_search_result(score: float, reranker_score: float | None = None) -> MagicMock:
    r = MagicMock()
    r.__getitem__ = MagicMock(side_effect=lambda k: {
        "chunk_id":    "doc1_p1_c0",
        "doc_id":      "doc1",
        "filename":    "test.pdf",
        "page_number": 1,
        "text":        "Some content here.",
    }[k])
    r.__contains__ = MagicMock(return_value=True)
    r.get = MagicMock(side_effect=lambda k, d=None: {
        "@search.score":         score,
        "@search.reranker_score": reranker_score,
    }.get(k, d))
    # Mimic dict-like access for score fields
    r.__getitem__.side_effect = lambda k: {
        "chunk_id":              "doc1_p1_c0",
        "doc_id":                "doc1",
        "filename":              "test.pdf",
        "page_number":           1,
        "text":                  "Some content here.",
        "@search.score":         score,
        "@search.reranker_score": reranker_score,
    }[k]
    return r


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_returns_structured_chunks(self):
        fake_results = [_make_search_result(2.5, reranker_score=3.1)]

        with (
            patch("retrieval.hybrid_search.embed_single_text", new=AsyncMock(return_value=[0.1] * 1536)),
            patch("retrieval.hybrid_search.get_search_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.search.return_value = iter(fake_results)
            mock_get_client.return_value = mock_client

            from retrieval.hybrid_search import hybrid_search
            results = await hybrid_search("primary endpoint")

        assert len(results) == 1
        chunk = results[0]
        assert chunk["chunk_id"] == "doc1_p1_c0"
        assert chunk["filename"] == "test.pdf"
        assert chunk["page_number"] == 1
        assert "text" in chunk
        assert "score" in chunk

    @pytest.mark.asyncio
    async def test_applies_doc_id_filter(self):
        with (
            patch("retrieval.hybrid_search.embed_single_text", new=AsyncMock(return_value=[0.0] * 1536)),
            patch("retrieval.hybrid_search.get_search_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.search.return_value = iter([])
            mock_get_client.return_value = mock_client

            from retrieval.hybrid_search import hybrid_search
            await hybrid_search("query", doc_id="specific-doc")

        call_kwargs = mock_client.search.call_args.kwargs
        assert "specific-doc" in call_kwargs.get("filter", "")

    @pytest.mark.asyncio
    async def test_no_filter_when_doc_id_is_none(self):
        with (
            patch("retrieval.hybrid_search.embed_single_text", new=AsyncMock(return_value=[0.0] * 1536)),
            patch("retrieval.hybrid_search.get_search_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.search.return_value = iter([])
            mock_get_client.return_value = mock_client

            from retrieval.hybrid_search import hybrid_search
            await hybrid_search("query", doc_id=None)

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs.get("filter") is None

    @pytest.mark.asyncio
    async def test_prefers_reranker_score_over_bm25_score(self):
        fake_results = [_make_search_result(0.5, reranker_score=3.8)]

        with (
            patch("retrieval.hybrid_search.embed_single_text", new=AsyncMock(return_value=[0.0] * 1536)),
            patch("retrieval.hybrid_search.get_search_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.search.return_value = iter(fake_results)
            mock_get_client.return_value = mock_client

            from retrieval.hybrid_search import hybrid_search
            results = await hybrid_search("query")

        # reranker score (3.8) should be used, not BM25 (0.5)
        assert results[0]["score"] == 3.8


class TestRetrievalQualityGate:
    def test_passes_when_above_threshold(self, sample_chunks):
        from retrieval.hybrid_search import retrieval_quality_gate
        sample_chunks[0]["score"] = 1.5
        assert retrieval_quality_gate(sample_chunks, threshold=0.5) is True

    def test_fails_when_below_threshold(self, sample_chunks):
        from retrieval.hybrid_search import retrieval_quality_gate
        sample_chunks[0]["score"] = 0.1
        assert retrieval_quality_gate(sample_chunks, threshold=0.5) is False

    def test_fails_for_empty_chunks(self):
        from retrieval.hybrid_search import retrieval_quality_gate
        assert retrieval_quality_gate([], threshold=0.5) is False

    def test_boundary_at_threshold(self, sample_chunks):
        from retrieval.hybrid_search import retrieval_quality_gate
        sample_chunks[0]["score"] = 0.5
        # exactly at threshold should pass
        assert retrieval_quality_gate(sample_chunks, threshold=0.5) is True
