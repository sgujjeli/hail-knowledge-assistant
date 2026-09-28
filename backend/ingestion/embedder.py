"""
Embedding generation — Azure OpenAI text-embedding-3-small.

Exact endpoint called by the SDK:
  POST https://{resource}.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings
       ?api-version=2024-02-01

Output: 1536-dimensional float vector per text string.

Batching: Azure OpenAI supports up to 2048 inputs per request.
We batch in groups of 100 to stay well within limits and handle
rate throttling gracefully.
"""
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from core.config import settings


def _build_embed_client() -> AsyncAzureOpenAI:
    """Mirror the same auth pattern as llm.py — key if set, else DefaultAzureCredential."""
    if settings.azure_openai_api_key:
        return AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AsyncAzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


# Single shared async client — thread-safe, reuse across requests
_client = _build_embed_client()

EMBEDDING_BATCH_SIZE = 100   # chunks per API call


async def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for all chunks.
    Mutates each chunk dict in-place, adding an 'embedding' key.

    Args:
        chunks: list of chunk dicts from chunker.chunk_text

    Returns:
        same list with 'embedding': list[float] added to each chunk
    """
    for batch_start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]

        response = await _client.embeddings.create(
            model=settings.azure_openai_embedding_deployment,
            input=texts,
        )

        for i, embedding_obj in enumerate(response.data):
            batch[i]["embedding"] = embedding_obj.embedding  # list of 1536 floats

    return chunks


async def embed_single_text(text: str) -> list[float]:
    """
    Embed a single query string for retrieval.
    Used by hybrid_search.py at query time.
    """
    response = await _client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=[text],
    )
    return response.data[0].embedding
