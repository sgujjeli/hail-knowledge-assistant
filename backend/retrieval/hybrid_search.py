"""
Hybrid search — combines BM25 keyword + vector similarity + semantic reranking
in a single Azure AI Search call (Reciprocal Rank Fusion).

Why hybrid?
  - BM25 catches exact keyword matches (error codes, product names, IDs)
  - Vector similarity catches semantic meaning ("slow response" ≈ "latency issue")
  - Semantic reranker re-scores top K with a cross-encoder for precision

Endpoint called by the SDK:
  POST https://{resource}.search.windows.net/indexes/{index}/docs/search
       ?api-version=2024-05-01-preview
"""
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from ingestion.indexer import get_search_client
from ingestion.embedder import embed_single_text
from core.config import settings

TOP_K = 5                   # chunks returned to the LLM
VECTOR_FIELDS = ["embedding"]


async def hybrid_search(
    query: str,
    doc_id: str | None = None,   # optional: restrict to a single document
) -> list[dict]:
    """
    Run a hybrid BM25 + vector + semantic reranker query against Azure AI Search.

    Returns list of chunk dicts:
        [{ chunk_id, doc_id, filename, page_number, text, score }, ...]
    """
    # 1. Embed the query at runtime
    query_vector = await embed_single_text(query)

    # 2. Build vector query object
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=TOP_K * 2,          # fetch more before reranking
        fields=",".join(VECTOR_FIELDS),
        exhaustive=False,                        # use HNSW ANN (fast)
    )

    # 3. Optional document-level filter
    search_filter = f"doc_id eq '{doc_id}'" if doc_id else None

    # 4. Execute hybrid search
    client: SearchClient = get_search_client()
    results = client.search(
        search_text=query,                        # BM25 keyword leg
        vector_queries=[vector_query],            # vector leg
        filter=search_filter,
        select=["chunk_id", "doc_id", "filename", "page_number", "text"],
        query_type="semantic",                    # enable semantic reranker
        semantic_configuration_name="default-semantic",
        query_caption="extractive",
        query_answer="extractive",
        top=TOP_K,
    )

    chunks = []
    for result in results:
        chunks.append({
            "chunk_id":    result["chunk_id"],
            "doc_id":      result["doc_id"],
            "filename":    result["filename"],
            "page_number": result["page_number"],
            "text":        result["text"],
            "score":       result["@search.reranker_score"] or result["@search.score"],
        })

    return chunks


def retrieval_quality_gate(chunks: list[dict], threshold: float = 0.5) -> bool:
    """
    Return True if retrieval quality is sufficient to generate a grounded answer.
    Blocks generation (and prevents hallucination) when top-ranked chunk is below threshold.

    Semantic reranker score range: 0–4 (Azure AI Search)
    BM25 score: unbounded, context-dependent — use 0.01 as bare minimum.
    """
    if not chunks:
        return False
    top_score = chunks[0]["score"]
    return top_score >= threshold
