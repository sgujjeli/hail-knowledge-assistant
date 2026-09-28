"""
Shared pytest fixtures for the Hail Knowledge Assistant test suite.

All Azure / OpenAI / Redis calls are mocked — tests run fully offline.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ── Fake settings (avoids real .env) ────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Patch settings so no real credentials are required."""
    import sys
    from types import ModuleType

    # Build a minimal fake settings object
    fake = MagicMock()
    fake.azure_openai_endpoint = "https://fake.openai.azure.com/"
    fake.azure_openai_api_key = "fake-key"
    fake.azure_openai_api_version = "2024-02-01"
    fake.azure_openai_chat_deployment = "gpt-4o"
    fake.azure_openai_embedding_deployment = "text-embedding-3-small"
    fake.azure_search_endpoint = "https://fake.search.windows.net"
    fake.azure_search_admin_key = "fake-search-key"
    fake.azure_search_index_name = "hail-documents"
    fake.azure_storage_connection_string = "DefaultEndpointsProtocol=https;AccountName=fake;..."
    fake.azure_storage_container = "hail-documents"
    fake.azure_speech_key = "fake-speech-key"
    fake.azure_speech_region = "uksouth"
    fake.anthropic_api_key = ""
    fake.database_url = "postgresql+asyncpg://postgres:password@localhost:5432/hail_test"
    fake.redis_url = "redis://localhost:6379/0"

    # Inject before any module imports settings
    core_mod = ModuleType("core")
    config_mod = ModuleType("core.config")
    config_mod.settings = fake
    sys.modules.setdefault("core", core_mod)
    sys.modules["core.config"] = config_mod

    yield fake


# ── Reusable chunk factory ───────────────────────────────────────────────────
@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id":    "doc1_p1_c0",
            "doc_id":      "doc1",
            "filename":    "clinical_trial.pdf",
            "page_number": 1,
            "text":        "The primary endpoint was a 30% reduction in adverse events.",
            "chunk_index": 0,
            "char_count":  60,
        },
        {
            "chunk_id":    "doc1_p2_c0",
            "doc_id":      "doc1",
            "filename":    "clinical_trial.pdf",
            "page_number": 2,
            "text":        "Secondary endpoints included patient-reported outcomes at 12 weeks.",
            "chunk_index": 0,
            "char_count":  67,
        },
    ]


@pytest.fixture
def sample_chunks_with_embeddings(sample_chunks):
    for chunk in sample_chunks:
        chunk["embedding"] = [0.01] * 1536
    return sample_chunks
