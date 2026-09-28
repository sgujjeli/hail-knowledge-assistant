"""
Unit tests — retrieval/prompts.py

Tests:
  - build_context_block: includes filename and page number in each chunk header
  - build_context_block: chunk text is present in output
  - build_messages: system message contains the context block
  - build_messages: user message matches the query
  - extract_citations: parses [Filename, Page X] patterns correctly
  - extract_citations: deduplicates repeated citations
  - extract_citations: returns empty list when no citations found
  - extract_citations: matches chunk metadata for enrichment
"""
import pytest


class TestBuildContextBlock:
    def test_includes_filename_and_page(self, sample_chunks):
        from retrieval.prompts import build_context_block
        result = build_context_block(sample_chunks)
        assert "clinical_trial.pdf" in result
        assert "Page 1" in result
        assert "Page 2" in result

    def test_includes_chunk_text(self, sample_chunks):
        from retrieval.prompts import build_context_block
        result = build_context_block(sample_chunks)
        assert "primary endpoint" in result
        assert "Secondary endpoints" in result

    def test_chunk_numbering(self, sample_chunks):
        from retrieval.prompts import build_context_block
        result = build_context_block(sample_chunks)
        assert "Chunk 1" in result
        assert "Chunk 2" in result

    def test_empty_chunks_returns_empty_string(self):
        from retrieval.prompts import build_context_block
        assert build_context_block([]) == ""


class TestBuildMessages:
    def test_system_message_contains_context(self, sample_chunks):
        from retrieval.prompts import build_messages
        messages = build_messages("What is the primary endpoint?", sample_chunks)
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "primary endpoint" in system_msg["content"]

    def test_user_message_is_query(self, sample_chunks):
        from retrieval.prompts import build_messages
        messages = build_messages("What is the primary endpoint?", sample_chunks)
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["content"] == "What is the primary endpoint?"

    def test_messages_structure(self, sample_chunks):
        from retrieval.prompts import build_messages
        messages = build_messages("Test query", sample_chunks)
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_system_prompt_contains_only_context_instruction(self, sample_chunks):
        from retrieval.prompts import build_messages
        messages = build_messages("query", sample_chunks)
        system_content = next(m for m in messages if m["role"] == "system")["content"]
        assert "ONLY" in system_content or "only" in system_content


class TestExtractCitations:
    def test_parses_citation_pattern(self, sample_chunks):
        from retrieval.prompts import extract_citations
        text = "The trial had 30% reduction [clinical_trial.pdf, Page 1]."
        result = extract_citations(text, sample_chunks)
        assert len(result) == 1
        assert result[0]["filename"] == "clinical_trial.pdf"
        assert result[0]["page_number"] == 1

    def test_multiple_citations(self, sample_chunks):
        from retrieval.prompts import extract_citations
        text = (
            "Primary [clinical_trial.pdf, Page 1]. "
            "Secondary [clinical_trial.pdf, Page 2]."
        )
        result = extract_citations(text, sample_chunks)
        pages = [c["page_number"] for c in result]
        assert 1 in pages
        assert 2 in pages

    def test_deduplicates_repeated_citations(self, sample_chunks):
        from retrieval.prompts import extract_citations
        text = (
            "Repeated [clinical_trial.pdf, Page 1] "
            "and again [clinical_trial.pdf, Page 1]."
        )
        result = extract_citations(text, sample_chunks)
        assert len(result) == 1

    def test_no_citations_returns_empty(self, sample_chunks):
        from retrieval.prompts import extract_citations
        result = extract_citations("No citations here.", sample_chunks)
        assert result == []

    def test_enriches_with_chunk_id(self, sample_chunks):
        from retrieval.prompts import extract_citations
        text = "See [clinical_trial.pdf, Page 1]."
        result = extract_citations(text, sample_chunks)
        assert result[0]["chunk_id"] == "doc1_p1_c0"

    def test_unknown_source_chunk_id_is_none(self, sample_chunks):
        from retrieval.prompts import extract_citations
        text = "See [unknown_file.pdf, Page 99]."
        result = extract_citations(text, sample_chunks)
        assert result[0]["chunk_id"] is None
