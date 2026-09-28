# Hail Knowledge Assistant

Production-grade AI knowledge assistant — live at [hailoop.co.uk](https://hailoop.co.uk).

**Stack:** FastAPI · Azure OpenAI (GPT-4o) · Azure AI Search · Azure Blob Storage · Azure Speech · PostgreSQL · Redis · Next.js

---

## What this demonstrates

| Panel question | Where it's answered |
|---|---|
| Where do you store a PDF? | `blob_storage.py` → Azure Blob; record in PostgreSQL |
| How do you parse it? | `ingestion/parser.py` → PyMuPDF (text + images), python-docx |
| How do you chunk it? | `ingestion/chunker.py` → RecursiveCharacterTextSplitter (800/150) |
| What library for embeddings? | `ingestion/embedder.py` → `openai.AsyncAzureOpenAI` |
| What endpoint do you call? | `https://{resource}.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings` |
| Where do you store vectors? | `ingestion/indexer.py` → Azure AI Search (vector + keyword fields) |
| How do you retrieve? | `retrieval/hybrid_search.py` → VectorizedQuery + BM25 + semantic reranker |
| How do you ground the LLM? | `retrieval/prompts.py` → system prompt with retrieved context injected |
| How do you reduce hallucinations? | retrieval gate + temperature=0 + source citations |
| How do you evaluate RAG quality? | `evaluation/ragas_eval.py` → faithfulness, answer_relevancy, context_precision |

---

## Architecture

```
User uploads PDF/DOCX
        │
        ▼
Azure Blob Storage  ─────────────────────────────────────────────┐
        │                                                         │
        ▼                                                   PostgreSQL
  parser.py                                             (doc status, sessions)
  PyMuPDF / python-docx
  text + images (GPT-4o Vision)
        │
        ▼
  chunker.py
  RecursiveCharacterTextSplitter
  chunk_size=800, overlap=150
        │
        ▼
  embedder.py
  Azure OpenAI text-embedding-3-small
  POST /openai/deployments/text-embedding-3-small/embeddings
        │
        ▼
  indexer.py
  Azure AI Search
  vector field (1536-dim) + searchable text field
        │
User asks question
        │
        ▼
  hybrid_search.py
  VectorizedQuery (cosine) + BM25 keyword + semantic reranker
  → top-5 chunks with scores
        │
        ▼
  grounding.py
  validate retrieval quality → gate before LLM call
        │
        ▼
  prompts.py
  system prompt: "answer ONLY from context, cite sources"
  + retrieved chunks injected
        │
        ▼
  llm.py
  Azure OpenAI GPT-4o  OR  Claude Sonnet
  temperature=0, streaming=True
        │
        ▼
  FastAPI SSE endpoint → Next.js EventSource
  tokens stream to UI in real time
        │
        ▼
  Azure Speech (optional voice)
  STT: microphone → transcript → RAG pipeline
  TTS: response → JennyNeural streaming audio
        │
        ▼
  RAGAS Evaluation
  faithfulness / answer_relevancy / context_precision / context_recall
  CI gate: faithfulness < 0.80 → flag for review
```

---

## Quick start

```bash
# 1. Clone and configure
git clone https://github.com/yourusername/hail-knowledge-assistant
cp .env.example .env
# Fill in Azure credentials in .env

# 2. Start dependencies
docker-compose up postgres redis -d

# 3. Run backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# 4. Run frontend
cd frontend
npm install && npm run dev
```

Backend: http://localhost:8000  
API docs: http://localhost:8000/docs  
Frontend: http://localhost:3000

---

## Key endpoints

| Method | Endpoint | What it does |
|---|---|---|
| POST | `/documents/upload` | Upload PDF/DOCX → blob → parse → chunk → embed → index |
| GET | `/documents/{doc_id}/status` | Check ingestion status |
| POST | `/chat/stream` | SSE: hybrid search → grounded GPT-4o/Claude stream |
| POST | `/voice/transcribe` | Azure Speech STT → transcript |
| POST | `/voice/synthesise` | Text → Azure JennyNeural audio stream |
| POST | `/evaluation/run` | Run RAGAS eval on golden dataset |
| GET | `/health` | Health check |

---

## RAGAS evaluation

```bash
cd backend
python -m evaluation.ragas_eval --input golden_dataset.json --output results.json
```

Metrics: **faithfulness · answer_relevancy · context_precision · context_recall**  
Threshold: faithfulness < 0.80 → `REVIEW_REQUIRED` flag

---

## Project structure

```
hail-knowledge-assistant/
├── backend/
│   ├── main.py                    # FastAPI app + lifespan
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── core/
│   │   ├── config.py              # pydantic-settings
│   │   ├── database.py            # SQLAlchemy async + asyncpg
│   │   └── redis_client.py
│   ├── models/
│   │   └── db_models.py           # Document, ChatSession ORM models
│   ├── ingestion/
│   │   ├── parser.py              # PyMuPDF PDF + python-docx + image extraction
│   │   ├── chunker.py             # RecursiveCharacterTextSplitter
│   │   ├── embedder.py            # Azure OpenAI text-embedding-3-small
│   │   ├── indexer.py             # Azure AI Search index creation + upload
│   │   └── blob_storage.py        # Azure Blob async upload
│   ├── retrieval/
│   │   └── hybrid_search.py       # VectorizedQuery + BM25 + semantic reranker
│   ├── generation/
│   │   ├── prompts.py             # System instructions + context injection
│   │   ├── llm.py                 # GPT-4o + Claude streaming clients
│   │   └── grounding.py           # Hallucination mitigation + citations
│   ├── evaluation/
│   │   └── ragas_eval.py          # RAGAS pipeline + CI threshold gate
│   └── api/routes/
│       ├── documents.py           # Upload + ingest endpoint
│       ├── chat.py                # SSE streaming chat
│       └── voice.py               # Azure Speech STT + TTS
├── frontend/
│   ├── app/
│   │   ├── page.tsx               # Main chat UI
│   │   └── upload/page.tsx
│   ├── components/
│   │   ├── ChatWindow.tsx
│   │   ├── DocumentUploader.tsx
│   │   └── VoiceInput.tsx
│   └── hooks/
│       └── useRAGChat.ts          # EventSource SSE hook
├── .env.example
└── docker-compose.yml
```
