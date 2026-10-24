# Multi-Modal RAG Knowledge Engine — Architecture Blueprint

> **Version:** 1.0  
> **Author:** Senior AI Architect  
> **Date:** 2026-10-24  
> **Status:** Awaiting Approval

---

## 1. Executive Summary

This document defines the complete architecture for a **Multi-Modal Retrieval-Augmented Generation (RAG) Knowledge Engine** — a production-grade system that ingests complex PDFs (text + visuals), builds a unified semantic index, and serves intelligent answers via a FastAPI backend. The system treats images as first-class citizens by converting them into dense text descriptions via a Vision-Language Model before embedding, enabling truly multi-modal retrieval from a single vector space.

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│              │     │   Ingestion Pipeline  │     │                  │
│  PDF Upload  ├────►│                       ├────►│   ChromaDB       │
│  (FastAPI)   │     │  Text ──► Chunker     │     │   (Unified       │
│              │     │  Images ──► VLM ──►   │     │    Collection)   │
└──────────────┘     │    Summaries          │     │                  │
                     └──────────────────────┘     └────────┬─────────┘
                                                           │
┌──────────────┐     ┌──────────────────────┐              │
│              │     │   RAG Pipeline        │◄─────────────┘
│  User Query  ├────►│                       │
│  (FastAPI)   │     │  Retrieve ──► Rank    │     ┌──────────────────┐
│              │     │  ──► Augment ──► LLM  ├────►│  Synthesized     │
└──────────────┘     │                       │     │  Answer + Image  │
                     └──────────────────────┘     │  References      │
                                                  └──────────────────┘
```

### 2.2 Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                         │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  /ingest    │  │  /query     │  │  /health   │  │  /documents│ │
│  │  endpoint   │  │  endpoint   │  │  endpoint  │  │  endpoint  │ │
│  └──────┬──────┘  └──────┬──────┘  └────────────┘  └────────────┘ │
│         │                │                                         │
│  ┌──────▼──────┐  ┌──────▼──────┐                                  │
│  │  Ingestion  │  │  Retrieval  │                                  │
│  │  Service    │  │  Service    │                                  │
│  └──────┬──────┘  └──────┬──────┘                                  │
│         │                │                                         │
│  ┌──────▼──────────────────────────────────────────────────────┐   │
│  │                     Core Services Layer                     │   │
│  │                                                             │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐ │   │
│  │  │ PDF Parser   │ │ Text Chunker │ │ Embedding Service   │ │   │
│  │  │ (PyMuPDF)    │ │ (Recursive)  │ │ (sentence-xformers) │ │   │
│  │  └──────────────┘ └──────────────┘ └─────────────────────┘ │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐ │   │
│  │  │ Image        │ │ VLM Client   │ │ VectorStore Client  │ │   │
│  │  │ Extractor    │ │ (Gemini/GPT) │ │ (ChromaDB)          │ │   │
│  │  └──────────────┘ └──────────────┘ └─────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Production-Grade Folder Structure

```
multimodal-rag-engine/
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory, lifespan, CORS
│   ├── config.py                  # Pydantic Settings (env-driven config)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py              # Top-level API router aggregator
│   │   ├── ingest.py              # POST /ingest — PDF upload endpoint
│   │   ├── query.py               # POST /query — RAG query endpoint
│   │   └── documents.py           # GET /documents — list ingested docs
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py          # PDF text extraction + image cropping
│   │   ├── text_chunker.py        # Recursive text splitting with overlap
│   │   ├── image_extractor.py     # Image extraction & preprocessing
│   │   └── exceptions.py          # Custom exception hierarchy
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py   # HuggingFace sentence-transformer wrapper
│   │   ├── vlm_service.py         # VLM API client (Gemini / GPT-4o-mini)
│   │   ├── vectorstore_service.py # ChromaDB CRUD operations
│   │   ├── ingestion_service.py   # Orchestrates full ingestion pipeline
│   │   └── retrieval_service.py   # Orchestrates retrieval + generation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── enums.py               # Enum definitions (ChunkType, etc.)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py          # File I/O helpers, path sanitization
│       └── logging_config.py      # Structured logging setup
│
├── data/
│   ├── uploads/                   # Raw uploaded PDFs
│   └── extracted_images/          # Cropped images from PDFs
│
├── chroma_db/                     # ChromaDB persistent storage
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared pytest fixtures
│   ├── test_pdf_parser.py
│   ├── test_text_chunker.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── scripts/
│   └── seed_sample.py             # Quick script to ingest a sample PDF
│
├── .env.example                   # Environment variable template
├── .gitignore
├── requirements.txt
├── README.md
└── plan.md                        # This document
```

---

## 4. Detailed Phase Breakdown

### Phase 1: Multi-Modal Ingestion Pipeline

**Goal:** Accept a PDF file, extract all text (page-by-page), chunk it intelligently, and crop/save all embedded images.

**Modules Delivered:**
| Module | Responsibility |
|---|---|
| `app/config.py` | Pydantic `Settings` class loading from `.env` |
| `app/core/pdf_parser.py` | Opens PDF via PyMuPDF, extracts raw text per page, discovers embedded images |
| `app/core/image_extractor.py` | Crops images from PDF pages, filters by minimum size, saves as PNG to `data/extracted_images/` |
| `app/core/text_chunker.py` | Recursive character-based text splitting with configurable `chunk_size` and `overlap` |
| `app/core/exceptions.py` | `PDFParsingError`, `ImageExtractionError`, `ChunkingError` |
| `app/models/schemas.py` | `TextChunk`, `ExtractedImage`, `IngestionResult` Pydantic models |
| `app/models/enums.py` | `ChunkType` enum (`TEXT`, `IMAGE_SUMMARY`) |
| `app/utils/file_utils.py` | `ensure_directory()`, `sanitize_filename()`, `generate_chunk_id()` |

**Key Design Decisions:**
- **Chunking Strategy:** Recursive splitting on `["\n\n", "\n", ". ", " "]` separators with 512-token chunks and 64-token overlap. This preserves paragraph boundaries.
- **Image Filtering:** Discard images smaller than 100×100 pixels (likely logos/icons, not informational charts).
- **Metadata Preservation:** Each `TextChunk` carries `source_pdf`, `page_number`, `chunk_index`. Each `ExtractedImage` carries `source_pdf`, `page_number`, `image_index`, `local_path`.

**Testable Output:** Given a PDF, the pipeline produces a list of `TextChunk` objects and a directory of extracted PNG images with a manifest.

---

### Phase 2: Semantic Image Processing

**Goal:** Send each extracted image to a Vision-Language Model and receive a dense textual summary describing the visual content.

**Modules Delivered:**
| Module | Responsibility |
|---|---|
| `app/services/vlm_service.py` | Async client wrapping the VLM API (Gemini 1.5 Flash or GPT-4o-mini). Handles base64 encoding, prompt construction, retries, and rate limiting. |

**VLM Prompt Engineering:**

```
You are a technical document analysis assistant. Describe the visual content 
of this image in rich, factual detail. Include:
1. The TYPE of visual (bar chart, line graph, flowchart, table, diagram, etc.)
2. ALL axis labels, legends, data values, and units visible.
3. Key trends, comparisons, or relationships the visual conveys.
4. Any title or caption text embedded in the image.

Output ONLY a dense descriptive paragraph. Do not include opinions or 
interpretations beyond what is visually present.
```

**Key Design Decisions:**
- **Async with Semaphore:** Limit concurrent VLM API calls (default: 5) to respect rate limits.
- **Retry with Exponential Backoff:** Up to 3 retries on transient API failures.
- **Fallback:** If VLM fails after retries, store a placeholder summary: `"[Image summary unavailable — visual content from page X]"`.
- **Cost Awareness:** Log token usage per image for monitoring.

**Testable Output:** Each `ExtractedImage` gains a `vlm_summary: str` field containing the generated description.

---

### Phase 3: Unified Vectorization

**Goal:** Embed all text chunks and image summaries into a single ChromaDB collection with rich metadata.

**Modules Delivered:**
| Module | Responsibility |
|---|---|
| `app/services/embedding_service.py` | Loads a `sentence-transformers` model (default: `all-MiniLM-L6-v2`), provides `embed_texts()` and `embed_single()` methods. Handles batching. |
| `app/services/vectorstore_service.py` | ChromaDB client wrapper. `add_documents()`, `query()`, `delete_collection()`, `list_documents()`. |
| `app/services/ingestion_service.py` | Top-level orchestrator: PDF → parse → chunk → extract images → VLM summarize → embed → store. |

**ChromaDB Collection Schema:**

```
Collection: "knowledge_base"

Document fields:
  - id:        str    — deterministic hash (source_pdf + page + chunk_index)
  - document:  str    — the text content (chunk text OR image VLM summary)
  - embedding: float[] — 384-dim vector (all-MiniLM-L6-v2)
  - metadata:
      - source_pdf:   str          — original filename
      - page_number:  int          — 1-indexed page
      - chunk_type:   str          — "text" | "image_summary"
      - chunk_index:  int          — ordering within page
      - image_path:   str | null   — local path to image (only for image_summary)
      - char_count:   int          — length of document text
      - created_at:   str          — ISO 8601 timestamp
```

**Key Design Decisions:**
- **Single Collection:** Both text and image summaries live in one collection. The `chunk_type` metadata field enables filtering if needed, but unified retrieval is the default.
- **Deterministic IDs:** Prevents duplicate ingestion of the same PDF.
- **Batch Embedding:** Process embeddings in batches of 64 to balance memory and throughput.
- **Model Loading:** Lazy-loaded singleton to avoid redundant model initialization across requests.

**Testable Output:** After ingestion, `chromadb.count()` reflects all chunks + image summaries. Metadata queries return correct `chunk_type` distributions.

---

### Phase 4: Retrieval & Generation API

**Goal:** Expose a query endpoint that retrieves relevant context from ChromaDB and synthesizes a final answer using the LLM.

**Modules Delivered:**
| Module | Responsibility |
|---|---|
| `app/services/retrieval_service.py` | Embeds the user query, queries ChromaDB for top-k results, constructs the augmented prompt, calls the LLM, and returns the structured response. |
| `app/api/ingest.py` | `POST /api/v1/ingest` — accepts PDF upload, triggers full pipeline. |
| `app/api/query.py` | `POST /api/v1/query` — accepts a question, returns answer + sources. |
| `app/api/documents.py` | `GET /api/v1/documents` — lists all ingested documents with stats. |
| `app/api/router.py` | Aggregates all route modules under `/api/v1`. |
| `app/main.py` | FastAPI app factory with lifespan handler, CORS, exception handlers. |
| `app/utils/logging_config.py` | Structured JSON logging configuration. |

**RAG Prompt Template:**

```
You are a knowledgeable assistant answering questions based on provided context.
Use ONLY the context below to answer the question. If the context is insufficient,
say so clearly — do not fabricate information.

Some context items are summaries of charts, graphs, or diagrams from the source
documents. When referencing visual data, mention that the information comes from
a figure or chart.

--- CONTEXT ---
{context_blocks}
--- END CONTEXT ---

Question: {user_query}

Provide a thorough, well-structured answer with specific data points from the context.
```

**API Contracts:**

#### `POST /api/v1/ingest`
```json
// Request: multipart/form-data with file field "pdf"
// Response 200:
{
  "status": "success",
  "document_id": "abc123",
  "filename": "report.pdf",
  "stats": {
    "total_pages": 12,
    "text_chunks": 47,
    "images_extracted": 8,
    "images_summarized": 8
  }
}
```

#### `POST /api/v1/query`
```json
// Request:
{
  "question": "What was the revenue trend in Q3?",
  "top_k": 5,
  "include_sources": true
}

// Response 200:
{
  "answer": "According to the Q3 financial report...",
  "sources": [
    {
      "content_preview": "Revenue grew 15% quarter-over-quarter...",
      "source_pdf": "q3_report.pdf",
      "page_number": 4,
      "chunk_type": "text",
      "relevance_score": 0.87,
      "image_path": null
    },
    {
      "content_preview": "Bar chart showing quarterly revenue...",
      "source_pdf": "q3_report.pdf",
      "page_number": 5,
      "chunk_type": "image_summary",
      "relevance_score": 0.82,
      "image_path": "/data/extracted_images/q3_report_p5_img0.png"
    }
  ],
  "metadata": {
    "model_used": "gemini-1.5-flash",
    "retrieval_time_ms": 45,
    "generation_time_ms": 1200,
    "total_context_tokens": 1850
  }
}
```

#### `GET /api/v1/documents`
```json
// Response 200:
{
  "documents": [
    {
      "document_id": "abc123",
      "filename": "q3_report.pdf",
      "pages": 12,
      "text_chunks": 47,
      "image_chunks": 8,
      "ingested_at": "2026-10-24T10:00:00Z"
    }
  ],
  "total_count": 1
}
```

**Key Design Decisions:**
- **Async End-to-End:** All I/O-bound operations (PDF parsing offloaded to thread pool, VLM calls, ChromaDB queries) use `async`/`await`.
- **Lifespan Handler:** Initializes embedding model and ChromaDB client on startup; cleans up on shutdown.
- **Static File Serving:** Mount `data/extracted_images/` so the frontend can render referenced charts.
- **Health Check:** `GET /api/v1/health` returns system status including ChromaDB connectivity and model load state.

---

## 5. Data Flow Walkthrough

```
User uploads PDF
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ INGESTION PIPELINE                                              │
│                                                                 │
│  1. Save PDF to data/uploads/                                   │
│  2. Open with PyMuPDF                                           │
│  3. For each page:                                              │
│     a. Extract raw text ──► TextChunk objects                   │
│     b. Extract images ──► filter by size ──► save as PNG        │
│  4. Chunk all text (recursive splitter, 512 chars, 64 overlap)  │
│  5. For each extracted image:                                   │
│     a. Encode to base64                                         │
│     b. Send to VLM with structured prompt                       │
│     c. Receive text summary ──► TextChunk (type=image_summary)  │
│  6. Embed ALL chunks (text + image summaries) via sentence-xf   │
│  7. Upsert into ChromaDB with metadata                         │
│  8. Return IngestionResult with stats                           │
└─────────────────────────────────────────────────────────────────┘

User sends query
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ RETRIEVAL + GENERATION PIPELINE                                 │
│                                                                 │
│  1. Embed user query with same sentence-transformer model       │
│  2. Query ChromaDB for top-k nearest neighbors                  │
│  3. Rank results by cosine similarity                           │
│  4. Build context window from retrieved chunks                  │
│  5. Construct augmented prompt (context + question)             │
│  6. Send to LLM for final answer synthesis                      │
│  7. Return answer + source metadata + image paths               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Configuration & Environment Variables

```env
# .env.example

# --- Application ---
APP_NAME=MultiModal-RAG-Engine
APP_VERSION=1.0.0
DEBUG=true
LOG_LEVEL=INFO

# --- VLM API ---
VLM_PROVIDER=gemini                     # "gemini" or "openai"
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key      # optional, if using GPT-4o-mini
VLM_MODEL_NAME=gemini-1.5-flash
VLM_MAX_CONCURRENT=5
VLM_MAX_RETRIES=3

# --- Embeddings ---
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=64

# --- ChromaDB ---
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# --- Chunking ---
CHUNK_SIZE=512
CHUNK_OVERLAP=64
MIN_IMAGE_DIMENSION=100

# --- Retrieval ---
DEFAULT_TOP_K=5

# --- Paths ---
UPLOAD_DIR=./data/uploads
EXTRACTED_IMAGES_DIR=./data/extracted_images
```

---

## 7. Error Handling Strategy

| Layer | Exception | HTTP Code | Behavior |
|---|---|---|---|
| PDF Parsing | `PDFParsingError` | 422 | Corrupt/encrypted PDF rejected with message |
| Image Extraction | `ImageExtractionError` | 200 (partial) | Log warning, continue without that image |
| VLM API | `VLMServiceError` | 200 (partial) | Retry 3×, then store placeholder summary |
| Embedding | `EmbeddingError` | 500 | Critical — fail the entire ingestion |
| ChromaDB | `VectorStoreError` | 500 | Critical — fail with details |
| Query (no results) | N/A | 200 | Return answer: "No relevant context found" |
| File Upload | `FileTooLarge` | 413 | Reject files > 50 MB |
| Validation | Pydantic errors | 422 | Auto-handled by FastAPI |

---

## 8. Performance Considerations

- **Embedding Model Singleton:** Loaded once at startup via lifespan, shared across requests. ~90 MB memory for `all-MiniLM-L6-v2`.
- **Thread Pool for CPU Work:** PDF parsing and image extraction run in `asyncio.to_thread()` to avoid blocking the event loop.
- **Batch Operations:** Embeddings computed in batches of 64. ChromaDB upserts batched.
- **Image Size Cap:** Images resized to max 1024px on longest edge before VLM submission to reduce API costs.

---

## 9. Future Enhancements (Post-MVP)

- [ ] Table extraction via `camelot` or `pdfplumber` structured parsing
- [ ] Hybrid search (keyword BM25 + semantic) via ChromaDB's built-in support
- [ ] Document deletion endpoint with cascade cleanup
- [ ] WebSocket streaming for long ingestion progress
- [ ] Frontend UI (React/Next.js) with image preview cards
- [ ] Docker containerization with `docker-compose.yml`
- [ ] Authentication via API keys or OAuth2
- [ ] CI/CD pipeline with GitHub Actions

---

## 10. Implementation Order

| Order | Deliverable | Dependencies |
|---|---|---|
| **Phase 1** | Config, PDF Parser, Image Extractor, Text Chunker, Schemas, Exceptions | None |
| **Phase 2** | VLM Service (Gemini/OpenAI client) | Phase 1 (images to summarize) |
| **Phase 3** | Embedding Service, VectorStore Service, Ingestion Orchestrator | Phase 1 + 2 |
| **Phase 4** | Retrieval Service, API Endpoints, App Factory, Logging | Phase 1 + 2 + 3 |

Each phase will be developed, reviewed, and tested before proceeding to the next.
