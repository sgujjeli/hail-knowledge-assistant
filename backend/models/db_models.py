from sqlalchemy import Column, String, Integer, DateTime, Text, BigInteger, Float
from sqlalchemy.sql import func
from core.database import Base
import uuid


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    blob_url = Column(Text, nullable=False)
    file_type = Column(String(10))                    # pdf | docx
    file_size_bytes = Column(BigInteger, default=0)
    status = Column(String(20), default="processing") # processing | ready | failed
    chunk_count = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String(36), nullable=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    model_used = Column(String(50))                   # gpt4o | claude
    chunks_retrieved = Column(Integer, default=0)
    top_retrieval_score = Column(Float, nullable=True)
    faithfulness_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
