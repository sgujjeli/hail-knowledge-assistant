"""
FastAPI application entry point — Hail Knowledge Assistant.

Architecture:
  Ingestion layer  → blob_storage → parser → chunker → embedder → indexer
  Retrieval layer  → hybrid_search (BM25 + vector + semantic reranker)
  Generation layer → prompts → llm (GPT-4o streaming / Claude fallback)
  Evaluation layer → ragas_eval (faithfulness, answer_relevancy)
  Voice layer      → Azure Speech STT + TTS

5-layer import DAG (enforced by import-linter in pre-commit):
  schemas → infrastructure → ai → services → api

FastAPI lifespan:
  startup  → create DB tables, ping Redis, create AI Search index
  shutdown → clean up connections
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import engine
from core.redis_client import redis_client
from models.db_models import Base
from ingestion.indexer import create_index_if_not_exists
from api.routes import documents, chat, voice


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — runs startup logic before first request, teardown on shutdown."""
    # ── Startup ──────────────────────────────────────────────────────────────
    # 1. Create Postgres tables (idempotent — won't destroy existing data)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Ping Redis
    await redis_client.ping()

    # 3. Create / update Azure AI Search index (idempotent)
    create_index_if_not_exists()

    print("✅ Hail Knowledge Assistant started — DB, Redis, AI Search ready")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    await engine.dispose()
    await redis_client.aclose()
    print("👋 Hail Knowledge Assistant shutdown complete")


app = FastAPI(
    title="Hail Knowledge Assistant",
    description=(
        "Production-grade RAG pipeline — hailoop.co.uk: "
        "Azure OpenAI GPT-4o · Azure AI Search hybrid · Azure Blob Storage · "
        "Azure Speech STT/TTS · FastAPI · PostgreSQL · Redis · RAGAS evaluation"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_swagger else None,
    redoc_url="/redoc" if settings.enable_swagger else None,
    openapi_url="/openapi.json" if settings.enable_swagger else None,
)

# ── CORS — allow Next.js frontend ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://hailoop.co.uk", "https://hail-ka.netlify.app", "https://hail-knowledge-assistant.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(voice.router)


@app.get("/health")
async def health():
    """Liveness probe for Docker / Kubernetes."""
    return {"status": "ok", "service": "hail-knowledge-assistant"}


@app.get("/")
async def root():
    return {
        "name": "Hail Knowledge Assistant",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload":       "POST /api/documents/upload",
            "documents":    "GET  /api/documents",
            "doc_status":   "GET  /api/documents/{doc_id}/status",
            "chat":         "POST /api/chat  (SSE streaming)",
            "transcribe":   "POST /api/voice/transcribe",
            "synthesize":   "POST /api/voice/synthesize",
        },
    }
