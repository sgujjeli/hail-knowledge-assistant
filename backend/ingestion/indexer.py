"""
Azure AI Search — index creation and document upload.

Index schema:
  - chunk_id      (String, key)
  - doc_id        (String, filterable)       — filter by document
  - filename      (String, filterable)
  - page_number   (Int32, filterable, sortable)
  - text          (String, searchable)       — BM25 keyword search
  - embedding     (Collection(Single), 1536) — vector similarity search

Vector algorithm: HNSW (Hierarchical Navigable Small World)
  — approximate nearest-neighbour, sub-linear query time.

Semantic configuration: reranks top BM25+vector hits using a
cross-encoder model — Azure's hosted semantic ranker.
"""
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.identity import DefaultAzureCredential
from core.config import settings

VECTOR_DIMENSIONS = 1536   # text-embedding-3-small output size


def _credential() -> AzureKeyCredential | TokenCredential:
    """
    Use admin key if provided; otherwise fall back to DefaultAzureCredential.
    Requires 'Search Index Data Contributor' + 'Search Service Contributor' roles.
    """
    if settings.azure_search_admin_key:
        return AzureKeyCredential(settings.azure_search_admin_key)
    return DefaultAzureCredential()


def create_index_if_not_exists() -> None:
    """
    Idempotent — safe to call on every startup.
    Creates the index only if it doesn't already exist.
    """
    client = SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=_credential(),
    )

    fields = [
        SimpleField(name="chunk_id",    type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id",      type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="filename",    type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32,
                    filterable=True, sortable=True),
        SearchableField(name="text",    type=SearchFieldDataType.String,
                        analyzer_name="en.lucene"),          # BM25 with English analyser
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algo",
                parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"},
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo",
            )
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="default-semantic",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="text")]
                ),
            )
        ]
    )

    index = SearchIndex(
        name=settings.azure_search_index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    client.create_or_update_index(index)


def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=_credential(),
    )


async def upload_chunks(chunks: list[dict]) -> None:
    """
    Upload embedded chunks to Azure AI Search in batches of 100.
    Azure AI Search max batch size is 1000; 100 is a safe default.
    """
    client = get_search_client()

    documents = [
        {
            "chunk_id":    chunk["chunk_id"],
            "doc_id":      chunk["doc_id"],
            "filename":    chunk["filename"],
            "page_number": chunk["page_number"],
            "text":        chunk["text"],
            "embedding":   chunk["embedding"],   # list[float], 1536 dims
        }
        for chunk in chunks
    ]

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        client.upload_documents(documents=batch)


async def delete_document_chunks(doc_id: str) -> None:
    """Remove all chunks for a given document from the index."""
    client = get_search_client()
    results = client.search(
        search_text="*",
        filter=f"doc_id eq '{doc_id}'",
        select=["chunk_id"],
    )
    chunk_ids = [{"chunk_id": r["chunk_id"]} for r in results]
    if chunk_ids:
        client.delete_documents(documents=chunk_ids)
