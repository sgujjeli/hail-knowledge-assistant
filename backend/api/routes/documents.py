"""
Document management routes.

POST /api/documents/upload
  - Validates file type (PDF, DOCX only)
  - Uploads to Azure Blob Storage
  - Creates DB record with status=processing
  - Triggers background ingestion task (parse → chunk → embed → index)
  - Returns document ID immediately (non-blocking)

GET /api/documents
  - Lists all documents with status, chunk count, page count

DELETE /api/documents/{doc_id}
  - Removes DB record + blob + AI Search chunks
"""
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone

from core.database import get_db
from models.db_models import Document
from ingestion.blob_storage import upload_to_blob
from ingestion.parser import parse_pdf, parse_docx, caption_image_with_gpt4o
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_chunks
from ingestion.indexer import upload_chunks, delete_document_chunks
from core.redis_client import cache_ingestion_status, get_ingestion_status

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_TYPES = {
    "application/pdf":                                                  "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}
MAX_FILE_SIZE = 20 * 1024 * 1024   # 20 MB


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload and ingest a PDF or DOCX document."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload PDF or DOCX only.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")

    doc_id = str(uuid.uuid4())
    blob_name = f"{doc_id}/{file.filename}"

    # Upload to Azure Blob
    blob_url = await upload_to_blob(file_bytes, blob_name, file.content_type)

    # Persist DB record
    doc = Document(
        id=doc_id,
        filename=file.filename,
        blob_url=blob_url,
        file_type=ALLOWED_TYPES[file.content_type],
        file_size_bytes=len(file_bytes),
        status="processing",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(doc)
    await db.commit()

    # Kick off background ingestion
    background_tasks.add_task(
        _ingest_document,
        doc_id=doc_id,
        filename=file.filename,
        file_bytes=file_bytes,
        file_type=ALLOWED_TYPES[file.content_type],
    )

    return {"doc_id": doc_id, "filename": file.filename, "status": "processing"}


async def _ingest_document(
    doc_id: str, filename: str, file_bytes: bytes, file_type: str
) -> None:
    """Background task: parse → chunk → embed → index → update DB status."""
    from core.database import AsyncSessionLocal

    await cache_ingestion_status(doc_id, {"status": "parsing", "progress": 10})

    try:
        # 1. Parse
        if file_type == "pdf":
            parsed = parse_pdf(file_bytes)
            # Caption images with GPT-4o Vision
            for page in parsed["pages"]:
                for img in page.get("images", []):
                    caption = await caption_image_with_gpt4o(img["data"], img["ext"])
                    page["text"] += f"\n\n[Image description: {caption}]"
        else:
            parsed = parse_docx(file_bytes)

        await cache_ingestion_status(doc_id, {"status": "chunking", "progress": 30})

        # 2. Chunk
        chunks = chunk_text(parsed["pages"], doc_id, filename)

        await cache_ingestion_status(doc_id, {"status": "embedding", "progress": 50})

        # 3. Embed
        embedded = await embed_chunks(chunks)

        await cache_ingestion_status(doc_id, {"status": "indexing", "progress": 80})

        # 4. Index
        await upload_chunks(embedded)

        # 5. Update DB
        async with AsyncSessionLocal() as session:
            doc = await session.get(Document, doc_id)
            doc.status = "ready"
            doc.chunk_count = len(chunks)
            doc.page_count = parsed["total_pages"]
            doc.updated_at = datetime.now(timezone.utc)
            await session.commit()

        await cache_ingestion_status(doc_id, {
            "status": "ready",
            "progress": 100,
            "chunk_count": len(chunks),
            "page_count": parsed["total_pages"],
        })

    except Exception as e:
        async with AsyncSessionLocal() as session:
            doc = await session.get(Document, doc_id)
            doc.status = "failed"
            doc.error_message = str(e)[:500]
            doc.updated_at = datetime.now(timezone.utc)
            await session.commit()

        await cache_ingestion_status(doc_id, {"status": "failed", "error": str(e)[:200]})


@router.get("")
async def list_documents(db: AsyncSession = Depends(get_db)):
    """Return all documents with their current ingestion status."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return [
        {
            "doc_id":      d.id,
            "filename":    d.filename,
            "status":      d.status,
            "chunk_count": d.chunk_count,
            "page_count":  d.page_count,
            "file_size":   d.file_size_bytes,
            "created_at":  d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.get("/{doc_id}/status")
async def get_document_status(doc_id: str):
    """Return real-time ingestion progress from Redis cache."""
    status = await get_ingestion_status(doc_id)
    if not status:
        raise HTTPException(status_code=404, detail="Document not found")
    return status


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a document from DB and AI Search index."""
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await delete_document_chunks(doc_id)
    await db.delete(doc)
    await db.commit()

    return {"deleted": doc_id}
