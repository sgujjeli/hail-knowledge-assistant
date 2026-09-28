"""
Chat route — SSE streaming RAG response.

POST /api/chat
  Request:  { query: str, doc_id?: str }
  Response: text/event-stream (Server-Sent Events)

SSE wire format:
  data: <token>\n\n          — LLM token
  data: [CITATIONS]...\n\n   — JSON citations after generation completes
  data: [DONE]\n\n           — stream end marker

Why SSE over WebSocket?
  - Simpler: unidirectional (server → client) is all we need
  - HTTP/1.1 compatible — no upgrade handshake
  - FastAPI StreamingResponse is native

Hallucination mitigations applied here:
  1. retrieval_quality_gate — block if top chunk score < threshold
  2. ONLY-use-context system prompt (in prompts.py)
  3. temperature=0 (in llm.py)
  4. RAGAS faithfulness check AFTER generation (async, non-blocking)
"""
import json
import asyncio
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.redis_client import cache_chat_response, get_cached_response
from retrieval.hybrid_search import hybrid_search, retrieval_quality_gate
from retrieval.prompts import build_messages, extract_citations, NO_CONTEXT_RESPONSE
from retrieval.llm import stream_with_fallback, generate_non_streaming
from evaluation.ragas_eval import evaluate_response
from models.db_models import ChatSession
from datetime import datetime, timezone

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    doc_id: str | None = None


@router.post("")
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Stream a grounded RAG answer over SSE.
    """
    # 1. Check Redis cache
    cache_key = f"{request.doc_id or 'all'}:{request.query}"
    cached = await get_cached_response(cache_key)
    if cached:
        async def _cached_stream():
            yield f"data: {cached['response']}\n\n"
            yield f"data: [CITATIONS]{json.dumps(cached['citations'])}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_cached_stream(), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    # 2. Hybrid search
    chunks = await hybrid_search(request.query, doc_id=request.doc_id)

    # 3. Retrieval quality gate
    if not retrieval_quality_gate(chunks):
        async def _no_context_stream():
            yield f"data: {NO_CONTEXT_RESPONSE}\n\n"
            yield "data: [CITATIONS][]\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_no_context_stream(), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    # 4. Build prompt
    messages = build_messages(request.query, chunks)

    # 5. Stream response
    async def _rag_stream():
        full_response = []

        async for token in stream_with_fallback(messages):
            full_response.append(token)
            # SSE format: "data: <token>\n\n"
            yield f"data: {token}\n\n"

        response_text = "".join(full_response)

        # 6. Extract and emit citations
        citations = extract_citations(response_text, chunks)
        yield f"data: [CITATIONS]{json.dumps(citations)}\n\n"
        yield "data: [DONE]\n\n"

        # 7. Persist to DB + cache (fire-and-forget after stream ends)
        asyncio.create_task(_persist_and_evaluate(
            db=db,
            query=request.query,
            response=response_text,
            chunks=chunks,
            doc_id=request.doc_id,
            cache_key=cache_key,
            citations=citations,
        ))

    return StreamingResponse(
        _rag_stream(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Cache-Control": "no-cache",
        },
    )


async def _persist_and_evaluate(
    db, query, response, chunks, doc_id, cache_key, citations
):
    """Persist ChatSession to Postgres and run RAGAS evaluation."""
    ragas_scores = await evaluate_response(query, response, chunks)

    session_record = ChatSession(
        doc_id=doc_id or (chunks[0]["doc_id"] if chunks else None),
        query=query,
        response=response,
        model_used="gpt-4o",
        chunks_retrieved=len(chunks),
        top_retrieval_score=chunks[0]["score"] if chunks else 0.0,
        faithfulness_score=ragas_scores["faithfulness"],
        created_at=datetime.now(timezone.utc),
    )
    db.add(session_record)
    await db.commit()

    # Cache the successful response
    await cache_chat_response(cache_key, {
        "response": response,
        "citations": citations,
        "faithfulness": ragas_scores["faithfulness"],
    })
