# Multi-Modal RAG Knowledge Engine — Codebase & File Architecture Guide

> **Document Version:** 1.0  
> **Audience:** Senior Engineers, Hiring Managers, Technical Interviewers, and Contributors  
> **Purpose:** Exhaustive, file-by-file breakdown explaining **what** every file in the repository does, **why** it exists, and **how** it accomplishes its work under the hood.

---

## 📑 Table of Contents

1. [Architectural Overview & Data Flow](#1-architectural-overview--data-flow)
2. [Root Configuration & Environment Files](#2-root-configuration--environment-files)
3. [Application Entry & Factory (`app/`)](#3-application-entry--factory-app)
4. [Data Models & Enums (`app/models/`)](#4-data-models--enums-appmodels)
5. [Core Ingestion & Parsing Algorithms (`app/core/`)](#5-core-ingestion--parsing-algorithms-appcore)
6. [Service Layer & External Integrations (`app/services/`)](#6-service-layer--external-integrations-appservices)
7. [API Layer & HTTP Endpoints (`app/api/`)](#7-api-layer--http-endpoints-appapi)
8. [Utilities & Infrastructure (`app/utils/`)](#8-utilities--infrastructure-apputils)
9. [Demonstration & Operational Scripts (`scripts/`)](#9-demonstration--operational-scripts-scripts)
10. [Automated Test Suite (`tests/`)](#10-automated-test-suite-tests)
11. [Data & Storage Directories (`data/`, `chroma_db/`)](#11-data--storage-directories-data-chroma_db)

---

## 1. Architectural Overview & Data Flow

Before examining individual files, here is how they collaborate during runtime:

```
[Client Request]
       │
       ▼
app/main.py ──► app/api/router.py
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
POST /api/v1/ingest           POST /api/v1/query
(app/api/ingest.py)          (app/api/query.py)
       │                             │
       ▼                             ▼
app/services/ingestion_service.py  app/services/retrieval_service.py
       │                             │
       ├─► app/core/pdf_parser.py    ├─► app/services/embedding_service.py
       │     ├─► text_chunker.py     │     (Embed user query vector)
       │     └─► image_extractor.py  │
       │                             ├─► app/services/vectorstore_service.py
       ├─► app/services/vlm_service.py     (Top-K cosine similarity search)
       │     (Dense visual summary)  │
       │                             └─► LLM Synthesis (Gemini/OpenAI)
       ├─► app/services/embedding_service.py
       │     (384-dim dense vectors)
       │
       └─► app/services/vectorstore_service.py
             (Upsert to ChromaDB unified collection)
```

---

## 2. Root Configuration & Environment Files

### `requirements.txt`
- **What it is:** Production and development dependency manifest.
- **Why it exists:** Pins exact package versions to guarantee reproducible builds across environments without version drift.
- **How it works:**
  - Categorized into functional layers: Core Backend (`fastapi`, `uvicorn`, `pydantic`), PDF Processing (`PyMuPDF`, `Pillow`), Embeddings & ML (`sentence-transformers`, `torch`, `onnxruntime`), Vector DB (`chromadb`), VLM APIs (`google-generativeai`, `openai`), Utilities (`httpx`, `tenacity`), and Testing (`pytest`).

### `.env.example`
- **What it is:** Configuration blueprint and documentation for environment variables.
- **Why it exists:** Prevents developers from accidentally committing API keys to Git while providing clear defaults for runtime settings.
- **How it works:** Defines all environment parameters with sensible defaults (e.g. `VLM_PROVIDER=gemini`, `EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2`, `CHUNK_SIZE=512`, `CHUNK_OVERLAP=64`). Developers copy this to `.env`.

### `.gitignore`
- **What it is:** Git exclusion rules specification.
- **Why it exists:** Prevents secrets, cached binaries, local databases, and temporary files from being tracked in version control.
- **How it works:** Filters out `__pycache__/`, `.env`, `venv/`, `.pytest_cache/`, `chroma_db/`, `.cache/`, and dynamically uploaded PDF/image assets in `data/uploads/*` and `data/extracted_images/*` (while preserving directory scaffolding via `.gitkeep`).

### `pytest.ini`
- **What it is:** Pytest runner configuration.
- **Why it exists:** Automatically registers the repository root on `sys.path` and configures plugin behavior.
- **How it works:** Sets `pythonpath = .` and `addopts = -p no:asyncio -p no:anyio` to avoid event loop conflicts on Windows during test discovery, and suppresses non-critical third-party deprecation warnings.

### `plan.md`
- **What it is:** The foundational architectural blueprint document.
- **Why it exists:** Outlines system architecture, data models, API contracts, error handling strategies, and implementation phases established before writing code.

### `README.md`
- **What it is:** Professional portfolio documentation for ML engineering reviewers.
- **Why it exists:** Provides recruiters and hiring managers with a clear summary of system capabilities, Mermaid architecture flowcharts, quickstart steps, and curl API examples.

---

## 3. Application Entry & Factory (`app/`)

### `app/__init__.py`
- **What it is:** Package initializer.
- **Why it exists:** Marks `app` as a Python package and defines `__version__ = "0.1.0"`.

### `app/config.py`
- **What it is:** Centralized configuration management using Pydantic Settings.
- **Why it exists:** Enforces type validation on environment variables at boot time and eliminates scattered `os.getenv()` calls.
- **How it works:**
  - Defines the `Settings` class inheriting from `pydantic_settings.BaseSettings`.
  - Configures `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`.
  - Exposes an `@lru_cache()` decorated function `get_settings()` returning a singleton `Settings` instance for high-throughput injection.

### `app/main.py`
- **What it is:** FastAPI application factory, lifespan controller, and server entrypoint.
- **Why it exists:** Bootstraps middleware, static asset routing, API routes, and handles startup/shutdown resource lifecycles.
- **How it works:**
  - **`lifespan(app: FastAPI)`**: Async context manager executed on server boot. Configures structured logging, ensures disk directories exist, and warms up the vector database and embedding engine singletons to prevent first-request cold starts.
  - **`create_application()`**: Factory function configuring:
    1. CORS middleware (`allow_origins=["*"]`) for frontend accessibility.
    2. Static file mounting (`/static/images`) pointing to `data/extracted_images/` so retrieved charts can be rendered in browsers.
    3. Aggregated API routes via `/api/v1`.
    4. Diagnostic endpoints: `GET /` and `GET /health` (reporting vector counts, active embedding engine, and VLM status).

---

## 4. Data Models & Enums (`app/models/`)

### `app/models/__init__.py`
- **What it is:** Namespace exporter for models.
- **Why it exists:** Simplifies imports across the project (e.g. `from app.models import TextChunk, ExtractedImage`).

### `app/models/enums.py`
- **What it is:** Enumerated constants.
- **Why it exists:** Prevents string typos and guarantees consistency across database queries and service boundaries.
- **How it works:**
  - `ChunkType(str, Enum)`: `"text"` vs `"image_summary"`. Used in ChromaDB metadata filtering to distinguish between paragraphs and figure descriptions.
  - `VLMProvider(str, Enum)`: `"gemini"` vs `"openai"`. Identifies the Vision-Language Model provider.

### `app/models/schemas.py`
- **What it is:** Pydantic v2 schemas for data validation and API serialization.
- **Why it exists:** Guarantees strict runtime typing and generates OpenAPI/Swagger schemas.
- **How it works:**
  - **`TextChunk`**: Represents an extracted text segment with `chunk_id`, `text`, `source_pdf`, `page_number`, `chunk_index`, and `char_count`.
  - **`ExtractedImage`**: Represents a visual asset with `image_id`, `source_pdf`, `page_number`, `local_path`, `width`, `height`, and `vlm_summary`.
  - **`QueryRequest`**: Validates user questions, `top_k` (1 to 20), `include_sources`, and optional `filter_source`.
  - **`SourceNode`**: Represents a citation returned to the user, including relevance score (cosine similarity), content snippet, and browser URL to the cropped figure.
  - **`QueryResponse`**: Combines user question, synthesized LLM answer, source citations, and latency telemetry (`retrieval_time_ms`, `generation_time_ms`).
  - **`HealthResponse`**, **`IngestionResult`**, **`DocumentListResponse`**: Response envelopes for REST endpoints.

---

## 5. Core Ingestion & Parsing Algorithms (`app/core/`)

### `app/core/__init__.py`
- **What it is:** Package initializer exporting core modules (`PDFParser`, `ImageExtractor`, `RecursiveTextChunker`, exceptions).

### `app/core/exceptions.py`
- **What it is:** Domain exception hierarchy.
- **Why it exists:** Enables granular error handling and maps domain failures to precise HTTP status codes.
- **How it works:**
  - Base: `RAGEngineException(Exception)`
  - Subclasses: `PDFParsingError`, `ImageExtractionError`, `ChunkingError`, `VLMServiceError`, `EmbeddingError`, `VectorStoreError`.

### `app/core/text_chunker.py`
- **What it is:** Intelligent recursive character text chunker with sliding window overlap.
- **Why it exists:** Naive character splitting splits sentences and words mid-token. This chunker respects grammatical and paragraph boundaries.
- **How it works:**
  - Operates hierarchically across separators: `["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]`.
  - Splits text using the highest-priority separator available.
  - Recursively splits any segment exceeding `chunk_size` (default: 512 chars).
  - Merges smaller segments together with sliding overlap (default: 64 chars) to preserve context across chunk boundaries.
  - Generates structured `TextChunk` instances with deterministic IDs (`doc_p1_c0`).

### `app/core/image_extractor.py`
- **What it is:** Visual figure extractor and preprocessor using PyMuPDF and Pillow.
- **Why it exists:** PDFs contain embedded figures, charts, and diagrams, but also hundreds of tiny UI icons, bullet points, and decorative rules that degrade retrieval if indexed.
- **How it works:**
  - Iterates over page image references via PyMuPDF (`page.get_images()`).
  - Extracts image byte stream using `doc.extract_image(xref)`.
  - Applies a dimension filter: discards images where `width < min_dimension` or `height < min_dimension` (default: 100px).
  - Normalizes color spaces (converting CMYK/Palette modes to standard RGB).
  - Constrains maximum dimensions (default: 2048px) using high-quality Lanczos resampling.
  - Persists images as PNGs to `data/extracted_images/<sanitized_doc_name>/` and returns typed `ExtractedImage` instances.

### `app/core/pdf_parser.py`
- **What it is:** Multi-modal PDF ingestion coordinator.
- **Why it exists:** Combines text extraction and visual figure extraction into a single, cohesive parsing step.
- **How it works:**
  - Opens PDF documents from disk paths or raw in-memory byte buffers via PyMuPDF.
  - Checks for encryption and empty page conditions.
  - Iterates page by page: extracts text and feeds it to `RecursiveTextChunker`; extracts visual figures and feeds them to `ImageExtractor`.
  - **`parse_async()`**: Uses `asyncio.to_thread()` to offload CPU-intensive PDF rendering and image extraction away from FastAPI's asynchronous event loop.

---

## 6. Service Layer & External Integrations (`app/services/`)

### `app/services/__init__.py`
- **What it is:** Package initializer exporting `EmbeddingService`, `VLMService`, `VectorStoreService`, `IngestionService`, and `RetrievalService`.

### `app/services/embedding_service.py`
- **What it is:** Dense vector embedding engine.
- **Why it exists:** Converts natural language text and image descriptions into 384-dimensional semantic vectors.
- **How it works:**
  - **Dual-Engine Architecture:**
    1. *Primary Engine:* Attempts to load Hugging Face `sentence-transformers` (`all-MiniLM-L6-v2`).
    2. *Resilient Fallback Engine:* If PyTorch encounters OS security blocks (e.g. Windows AppLocker `WinError 4551` blocking `torch_python.dll`), it automatically switches to ChromaDB's ONNX Runtime engine (`onnxruntime`), which runs the identical model weights without PyTorch dependencies.
  - Provides `embed_texts()` with batching (default: 64) and async wrappers (`embed_texts_async()`).

### `app/services/vlm_service.py`
- **What it is:** Vision-Language Model interface supporting Google Gemini and OpenAI.
- **Why it exists:** Translates non-searchable visual graphics (bar charts, line graphs, system architectures) into dense, highly searchable factual descriptions.
- **How it works:**
  - **Prompt Engineering:** Enforces `VLM_SYSTEM_PROMPT` instructing the model to describe visual type, axis labels, legends, units, data points, and factual trends without speculative preamble.
  - **Concurrency Control:** Wraps API requests in `asyncio.Semaphore(vlm_max_concurrent)` (default: 5) to prevent rate limit (HTTP 429) errors during bulk document ingestion.
  - **Retry Policy:** Uses `@retry` (`tenacity`) with exponential backoff (2s to 10s) across up to 3 attempts.
  - **Graceful Fallback:** If API keys are missing or calls fail repeatedly, generates a structured synthetic fallback description rather than crashing the pipeline.

### `app/services/vectorstore_service.py`
- **What it is:** Persistent vector database manager interfacing with ChromaDB.
- **Why it exists:** Provides persistent storage, cosine similarity indexing, and metadata filtering for multi-modal vectors.
- **How it works:**
  - Initializes `chromadb.PersistentClient` pointing to `chroma_db/`.
  - Configures the `knowledge_base` collection with HNSW cosine distance (`{"hnsw:space": "cosine"}`).
  - **`upsert_documents()`**: Performs idempotent upserts of IDs, document texts, metadatas, and vector embeddings.
  - **`query()`**: Executes vector similarity searches with optional metadata filters (e.g. `where={"chunk_type": "image_summary"}`).
  - **`delete_by_source()`**: Performs cascade deletions of all vectors originating from a specified PDF.
  - **`list_ingested_documents()`**: Scans collection metadata to return a document catalog with text/image counts.

### `app/services/ingestion_service.py`
- **What it is:** End-to-end multi-modal ingestion pipeline orchestrator.
- **Why it exists:** Coordinates the entire multi-stage ingestion workflow: parsing → visual summarization → embedding → vector storage.
- **How it works:**
  1. Calls `PDFParser.parse_async()` to extract text chunks and visual figures.
  2. Calls `VLMService.summarize_extracted_images()` concurrently to attach dense summaries to images.
  3. Formats unified vector records for both text chunks (`chunk_type="text"`) and image summaries (`chunk_type="image_summary"` with `image_path`).
  4. Generates dense embeddings via `EmbeddingService.embed_texts_async()`.
  5. Upserts all records into ChromaDB via `VectorStoreService.upsert_documents_async()`.
  6. Returns a typed `IngestionResult` summarizing processing statistics.

### `app/services/retrieval_service.py`
- **What it is:** RAG retrieval and answer synthesis orchestrator.
- **Why it exists:** Connects user queries to the vector database and formats grounded prompts for final LLM synthesis.
- **How it works:**
  1. Embeds the user question using `EmbeddingService`.
  2. Retrieves top-k nearest neighbors from ChromaDB.
  3. Converts cosine distance into a normalized relevance score (`1.0 - distance`).
  4. Generates browser-accessible image URLs (`/static/images/...`) for visual citations.
  5. Assembles context blocks into `RAG_SYSTEM_PROMPT_TEMPLATE`, instructing the LLM to ground its answer strictly on retrieved data and cite figures explicitly.
  6. Calls Gemini, OpenAI, or the built-in grounded synthesizer to generate the final response.

---

## 7. API Layer & HTTP Endpoints (`app/api/`)

### `app/api/__init__.py`
- **What it is:** Package initializer for API modules.

### `app/api/router.py`
- **What it is:** Central router aggregator.
- **Why it exists:** Consolidates all sub-routers under the `/api/v1` prefix for clean mounting into the main application.

### `app/api/ingest.py`
- **What it is:** PDF upload and ingestion endpoint.
- **Endpoints:**
  - `POST /api/v1/ingest`: Accepts `multipart/form-data` PDF uploads. Validates extension (`.pdf`), rejects 0-byte files, enforces a 50 MB size limit, and delegates to `IngestionService.ingest_bytes()`. Handles exceptions with typed HTTP 400, 413, 422, or 500 responses.

### `app/api/query.py`
- **What it is:** Natural language query and answer endpoint.
- **Endpoints:**
  - `POST /api/v1/query`: Accepts `QueryRequest` JSON. Calls `RetrievalService.query()`, returning synthesized answer, relevance scores, and referenced image URLs.

### `app/api/documents.py`
- **What it is:** Knowledge base document management endpoints.
- **Endpoints:**
  - `GET /api/v1/documents`: Returns a list of all indexed documents with text/image chunk counts and page counts.
  - `DELETE /api/v1/documents/{filename}`: Purges all indexed vectors and metadata for the specified document.

---

## 8. Utilities & Infrastructure (`app/utils/`)

### `app/utils/__init__.py`
- **What it is:** Package initializer exporting utility functions.

### `app/utils/file_utils.py`
- **What it is:** Filesystem and cryptographic identifier helpers.
- **Why it exists:** Ensures safe cross-platform file paths and deterministic identifiers.
- **How it works:**
  - `ensure_directory(path)`: Idempotently creates directory hierarchies.
  - `sanitize_filename(name)`: Strips unsafe characters and whitespace from filenames.
  - `generate_chunk_id(source_pdf, page, idx)`: Produces deterministic IDs like `report_p1_c0`.
  - `generate_image_id(source_pdf, page, idx)`: Produces deterministic IDs like `report_p2_img0`.
  - `generate_document_id(filename)`: Generates hash-backed document identifiers.

### `app/utils/logging_config.py`
- **What it is:** Production logging configuration.
- **Why it exists:** Replaces unformatted stdout prints with structured, timestamped log messages.
- **How it works:** Formats log records with timestamps, log levels, file names, line numbers, and messages, while suppressing verbose debug logging from third-party libraries (`chromadb`, `urllib3`, `httpx`).

---

## 9. Demonstration & Operational Scripts (`scripts/`)

### `scripts/seed_sample.py`
- **What it is:** Standalone demonstration and verification script.
- **Why it exists:** Allows developers and hiring managers to immediately run and verify the entire multi-modal pipeline without launching the FastAPI server or uploading files manually.
- **How it works:**
  1. Uses PyMuPDF and Pillow to synthesize a realistic 2-page PDF (`data/uploads/sample_quarterly_report.pdf`) containing text paragraphs and an embedded 500×300 RGB bar chart.
  2. Runs `PDFParser` to extract text and crop the chart.
  3. Runs `VLMService` to generate a dense semantic summary of the chart.
  4. Runs `EmbeddingService` and `VectorStoreService` to index all records into ChromaDB.
  5. Performs a trial multi-modal query (`"What was the quarterly model accuracy and latency metric?"`) and prints top retrieved contexts with referenced image asset paths.

---

## 10. Automated Test Suite (`tests/`)

The test suite contains **28 tests** across 7 test files, achieving 100% pass rates:

### `tests/__init__.py`
- **What it is:** Package initializer for tests.

### `tests/test_pdf_parser.py` (4 tests)
- **What it tests:**
  - Non-existent file rejection (`PDFParsingError`).
  - Empty byte buffer rejection (`PDFParsingError`).
  - Synthetic multi-page PDF parsing: verifies text extraction, large figure extraction (>100px), and filtering of small icons (<100px).
  - Asynchronous thread-pool parsing (`parse_async`).

### `tests/test_text_chunker.py` (5 tests)
- **What it tests:**
  - Parameter validation (rejects `chunk_size <= 0` or `chunk_overlap >= chunk_size`).
  - Empty and whitespace text handling.
  - Single-chunk creation for short text.
  - Paragraph boundary preservation across `\n\n` separators.
  - Sliding window overlap verification between consecutive chunks.

### `tests/test_vlm_service.py` (6 tests)
- **What it tests:**
  - Missing image file validation (`VLMServiceError`).
  - Synthetic fallback description generation when API keys are unconfigured.
  - Batch image summarization with concurrent `ExtractedImage` updates.
  - Mocked Gemini Vision API calls (verifying prompt structure and PIL Image passing).
  - Mocked OpenAI Vision API calls (verifying base64 data URL formatting).
  - Graceful degradation on API failure (ensures pipeline returns a fallback description rather than raising an uncaught exception).

### `tests/test_vectorstore.py` (5 tests)
- **What it tests:**
  - Initial collection creation and zero-count check.
  - Document upsert and vector count tracking.
  - Cosine distance similarity retrieval.
  - Metadata filtering (`where={"chunk_type": "image_summary"}`).
  - Cascade document deletion by `source_pdf`.

### `tests/test_embedding_service.py` (2 tests)
- **What it tests:**
  - Single text query embedding dimension (384 dimensions) and float normalization.
  - Batch text embedding preserving vector count.

### `tests/test_ingestion.py` (1 test)
- **What it tests:**
  - Full pipeline integration: Synthetic PDF → Parsing → Visual Extraction → VLM Summary → Dense Embedding → ChromaDB index verification.

### `tests/test_api.py` (5 tests)
- **What it tests:**
  - `GET /` operational status.
  - `GET /health` system diagnostics.
  - `POST /api/v1/ingest` rejection of non-PDF files (HTTP 422).
  - `POST /api/v1/ingest` rejection of empty files (HTTP 400).
  - Complete REST lifecycle: PDF upload → Document listing → Natural language query with visual source citations → Document deletion.

---

## 11. Data & Storage Directories (`data/`, `chroma_db/`)

### `data/uploads/`
- **Purpose:** Local filesystem directory storing raw uploaded PDF files before and during ingestion.

### `data/extracted_images/`
- **Purpose:** Organized storage directory holding cropped figures and charts (e.g. `data/extracted_images/<doc_name>/<doc_name>_p2_img0.png`). Mounted as a static file directory in FastAPI at `/static/images`.

### `chroma_db/`
- **Purpose:** On-disk persistent storage for ChromaDB containing HNSW vector indexes, document text, and metadata mappings.
