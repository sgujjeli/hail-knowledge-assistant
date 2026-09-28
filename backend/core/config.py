from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Azure OpenAI ──────────────────────────────────────────────────────
    # Endpoint: https://{resource}.openai.azure.com/
    # Full chat URL built by SDK:
    #   {endpoint}/openai/deployments/{chat_deployment}/chat/completions?api-version={version}
    # Full embedding URL:
    #   {endpoint}/openai/deployments/{embedding_deployment}/embeddings?api-version={version}
    azure_openai_endpoint: str
    azure_openai_api_key: str = ""   # leave blank to use Managed Identity / az login
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    # ── Azure AI Search ───────────────────────────────────────────────────
    # Endpoint: https://{resource}.search.windows.net
    azure_search_endpoint: str
    azure_search_admin_key: str = ""   # leave blank to use Managed Identity / az login
    azure_search_index_name: str = "hail-documents"

    # ── Azure Blob Storage ────────────────────────────────────────────────
    azure_storage_connection_string: str = ""  # leave blank to use Managed Identity / az login
    azure_storage_account_name: str = ""       # required when using Managed Identity
    azure_storage_container: str = "hail-documents"

    # ── Azure Speech ──────────────────────────────────────────────────────
    azure_speech_key: str
    azure_speech_region: str = "uksouth"

    # ── Anthropic (Claude Sonnet — fallback LLM) ─────────────────────────
    anthropic_api_key: str = ""
    enable_anthropic_fallback: bool = False   # set True in .env to activate

    # ── PostgreSQL ────────────────────────────────────────────────────────
    database_url: str  # postgresql+asyncpg://user:pass@host:5432/hail

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"


settings = Settings()
