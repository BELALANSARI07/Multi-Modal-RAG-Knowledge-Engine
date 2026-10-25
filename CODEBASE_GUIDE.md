# Multi-Modal RAG Knowledge Engine — Codebase & File Architecture Guide

> **Document Version:** 1.2  
> **Audience:** Senior Engineers, Hiring Managers, Technical Interviewers, and Contributors  
> **Purpose:** Exhaustive, file-by-file breakdown explaining **what** every file in the repository does, **why** it exists, and **how** it accomplishes its work under the hood.

---

## 📑 Table of Contents

1. [Architectural Overview & Data Flow](#1-architectural-overview--data-flow)
2. [Root Configuration, Packaging & Governance Files](#2-root-configuration-packaging--governance-files)
3. [Containerization & CI/CD Pipeline](#3-containerization--cicd-pipeline)
4. [Application Entry & Factory (`app/`)](#4-application-entry--factory-app)
5. [Data Models & Enums (`app/models/`)](#5-data-models--enums-appmodels)
6. [Core Ingestion & Parsing Algorithms (`app/core/`)](#6-core-ingestion--parsing-algorithms-appcore)
7. [Service Layer & External Integrations (`app/services/`)](#7-service-layer--external-integrations-appservices)
8. [API Layer, Middleware & Dependencies (`app/api/`)](#8-api-layer-middleware--dependencies-appapi)
9. [Utilities & Infrastructure (`app/utils/`)](#9-utilities--infrastructure-apputils)
10. [Demonstration & Operational Scripts (`scripts/`)](#10-demonstration--operational-scripts-scripts)
11. [Automated Test Suite (`tests/`)](#11-automated-test-suite-tests)
12. [Data & Storage Directories (`data/`, `chroma_db/`)](#12-data--storage-directories-data-chroma_db)

---

## 1. Architectural Overview & Data Flow

```
[Client Request]
       │
       ▼
RequestIDMiddleware (X-Request-ID Header Assignment)
       │
       ▼
app/main.py ──► app/api/router.py
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
POST /api/v1/ingest           POST /api/v1/query
(app/api/ingest.py)          (app/api/query.py)
  [Auth & Rate Limit Guards]    [Auth & Rate Limit Guards]
       │                             │
       ▼                             ▼
app/services/ingestion_service.py  app/services/retrieval_service.py
  (Transactional Rollback)      (Relevance & Context Guardrails)
       │                             │
       ├─► app/core/pdf_parser.py    ├─► app/services/embedding_service.py
       │     (Magic Header Check)    │     (Embed user query vector)
       │     ├─► text_chunker.py     │
       │     └─► image_extractor.py  ├─► app/services/vectorstore_service.py
       │           (SHA-256 Dedup)   │     (Top-K cosine similarity search)
       │                             │
       ├─► app/services/vlm_service.py └─► LLM Synthesis (Gemini / OpenAI)
       │     (Dense visual summary)
       │
       ├─► app/services/embedding_service.py
       │     (Dual-Engine: SentenceTransformer / ONNX Runtime)
       │
       └─► app/services/vectorstore_service.py
             (Dimension Validation + ChromaDB Upsert)
```

---

## 2. Root Configuration, Packaging & Governance Files

### `requirements.txt`
- **What it is:** Pinned production dependency manifest.
- **Why it exists:** Freezes exact package versions (`fastapi==0.115.0`, `PyMuPDF==1.24.10`, `chromadb==0.5.7`, etc.) to prevent supply chain drift.

### `requirements-dev.txt`
- **What it is:** Development and QA dependency manifest.
- **Why it exists:** Extends `requirements.txt` with test runners (`pytest`, `pytest-cov`), linters (`ruff`), and pre-commit hooks.

### `pyproject.toml`
- **What it is:** Modern PEP 621 Python project specification.
- **Why it exists:** Unifies package metadata, Python version declaration (`>=3.10`), Ruff lint rules (line length 120, strict import sorting), and Pytest configuration in a single standard configuration file.

### `.python-version`
- **What it is:** Version declaration file pinning Python `3.10.11` for pyenv, asdf, and mise.

### `.env.example`
- **What it is:** Configuration template for environment variables.
- **Why it exists:** Documents all configurable tunables (`VLM_PROVIDER`, `API_KEY`, `RATE_LIMIT_REQUESTS_PER_MINUTE`, `MIN_RELEVANCE_SCORE`, etc.) without exposing live secrets.

### `.gitignore`
- **What it is:** Git exclusion rules specification.
- **Why it exists:** Prevents secrets, cached binaries, local databases, and temporary files from being tracked in version control while preserving directory scaffolding via `.gitkeep`.

### `LICENSE`
- **What it is:** Official MIT License document with copyright notice.

### `SECURITY.md`
- **What it is:** Security vulnerability reporting policy and vulnerability response timelines.

### `CONTRIBUTING.md`
- **What it is:** Contribution guidelines, development environment setup, and PR checklist.

### `CODE_OF_CONDUCT.md`
- **What it is:** Contributor Covenant v2.1 community standard.

### `CHANGELOG.md`
- **What it is:** Project version history adhering to Keep a Changelog.

### `.gitleaks.toml`
- **What it is:** Secret-scanning configuration protecting against accidental credential commits.

---

## 3. Containerization & CI/CD Pipeline

### `Dockerfile`
- **What it is:** Multi-stage production container image.
- **Why it exists:** Provides consistent deployment across cloud environments.
- **How it works:**
  - Uses `python:3.10-slim` base image.
  - Installs system libraries for MuPDF and Pillow.
  - Creates and drops privileges to a dedicated non-root user (`appuser`, UID 1000).
  - Includes a `HEALTHCHECK` probe targeting `/health/live`.
  - Starts Uvicorn with `--proxy-headers` for reverse proxy compatibility.

### `docker-compose.yml`
- **What it is:** Multi-container orchestration definition.
- **Why it exists:** Allows one-command deployment (`docker compose up -d`) with persistent volumes (`chroma_data`, `upload_data`, `image_data`), CPU/RAM resource limits, and health monitoring.

### `.dockerignore`
- **What it is:** Docker build context exclusion list, omitting tests, caches, virtual environments, and `.git` folders.

### `.github/workflows/ci.yml`
- **What it is:** GitHub Actions continuous integration workflow.
- **Why it exists:** Automatically verifies code quality on every push and pull request to `main`.
- **How it works:**
  1. **Job 1 (Lint):** Runs `ruff check .` for style and import correctness.
  2. **Job 2 (Security Scan):** Executes `gitleaks` to detect exposed API keys or credentials.
  3. **Job 3 (Test Matrix):** Runs the full 37-test test suite across Python 3.10, 3.11, and 3.12 runners.

---

## 4. Application Entry & Factory (`app/`)

### `app/main.py`
- **What it is:** FastAPI application factory, lifespan controller, and server entrypoint.
- **Key Responsibilities:**
  - Configures `RequestIDMiddleware` for correlation tracking.
  - Configures restricted CORS with `cors_origins`.
  - Mounts static files at `/static/images` pointing to `data/extracted_images/`.
  - Global exception handler shielding internal tracebacks from API callers.
  - Diagnostic endpoints: `GET /`, `GET /health`, `GET /health/live`, and `GET /health/ready`.

### `app/config.py`
- **What it is:** Centralized configuration management using Pydantic Settings.
- **Key Responsibilities:**
  - `debug: bool = False` production default.
  - Strict validator for `vlm_provider` (`gemini` or `openai`).
  - Tunables for rate limiting, relevance score filtering, and context size guardrails.

---

## 5. Data Models & Enums (`app/models/`)

### `app/models/schemas.py`
- **What it is:** Pydantic v2 schemas for data validation and API serialization.
- **Key Models:**
  - `TextChunk`: Structured text segment with deterministic ID and metadata.
  - `ExtractedImage`: Visual figure record with dimensions, path, and `vlm_summary`.
  - `DocumentMetadata`: Persistent registry record with SHA-256 hash and size.
  - `QueryRequest`, `QueryResponse`, `SourceNode`: RAG query request and response contracts.
  - `HealthResponse`: System diagnostic payload.

### `app/models/enums.py`
- **What it is:** Typed enumerations (`ChunkType.TEXT`, `ChunkType.IMAGE_SUMMARY`, `VLMProvider.GEMINI`, `VLMProvider.OPENAI`).

---

## 6. Core Ingestion & Parsing Algorithms (`app/core/`)

### `app/core/pdf_parser.py`
- **What it is:** Multi-modal PDF ingestion coordinator.
- **Key Features:**
  - Validates `%PDF-` magic byte header before parsing.
  - Extracts text via `RecursiveTextChunker` and images via `ImageExtractor`.
  - Detects blank documents (0 text, 0 images) and raises `DocumentEmptyError`.
  - `parse_async()` offloads CPU-intensive MuPDF rendering to a thread pool.

### `app/core/image_extractor.py`
- **What it is:** Visual figure extractor and deduplicator using PyMuPDF and Pillow.
- **Key Features:**
  - Filters out icons/decorations smaller than 100px.
  - Computes SHA-256 hashes of image byte streams to prevent identical logos or figures from being processed multiple times.
  - Normalizes color modes to RGB and resizes oversized figures with Lanczos resampling.

### `app/core/text_chunker.py`
- **What it is:** Recursive character text chunker with sliding window overlap.
- **Key Features:**
  - Splits hierarchically across `["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]`.
  - Preserves grammatical boundaries within `chunk_size` (512 chars) and `chunk_overlap` (64 chars).

### `app/core/exceptions.py`
- **What it is:** Granular domain exception hierarchy rooted at `RAGEngineException`.

---

## 7. Service Layer & External Integrations (`app/services/`)

### `app/services/embedding_service.py`
- **What it is:** Dense vector embedding engine.
- **Key Features:**
  - Dual-engine design: Hugging Face `sentence-transformers` with automatic fallback to high-performance ONNX Runtime (`onnxruntime`), ensuring zero failure on Windows AppLocker/WDAC environments.

### `app/services/vlm_service.py`
- **What it is:** Vision-Language Model interface for Gemini 1.5 Flash and GPT-4o-mini.
- **Key Features:**
  - Strict system prompt extracting visual types, axis labels, legends, units, and data trends.
  - Bound by `asyncio.Semaphore(vlm_max_concurrent)` (default: 5).
  - Exponential backoff retry restricted strictly to transient network exceptions (`httpx.ConnectError`, `TimeoutError`, HTTP 429/503).
  - Timeout protection (`vlm_timeout_seconds = 30.0`).

### `app/services/vectorstore_service.py`
- **What it is:** ChromaDB persistent collection manager.
- **Key Features:**
  - Validates vector dimensions against expected model dimension before upserting.
  - Cosine distance HNSW indexing with metadata filtering (`chunk_type="image_summary"`).
  - Cascade deletion of all document chunks by source.

### `app/services/ingestion_service.py`
- **What it is:** Transactional ingestion pipeline orchestrator.
- **Key Features:**
  - Sanitize filenames and collision-resistant storage path generation.
  - Duplicate handling (`duplicate_strategy = "replace"` or `"skip"`).
  - Transactional rollback: purges uploaded PDF and extracted image directories if ChromaDB or embedding fails.
  - Thread-safe, atomic document registry writes at `data/documents_registry.json` using temporary files, `os.replace`, and recovery from corrupted JSON.
  - Comprehensive cascade delete (`delete_document_full`).

### `app/services/retrieval_service.py`
- **What it is:** Context retrieval, guardrails, and answer synthesis orchestrator.
- **Key Features:**
  - Filters out low-confidence matches using `min_relevance_score` (default: 0.20).
  - Truncates context blocks using `max_context_chars` (default: 8000) to prevent token window overflow.
  - Synthesizes grounded answers citing document page numbers and visual figures.
  - Exponential backoff retry on transient connection failures and timeout enforcement (`vlm_timeout_seconds`).

---

## 8. API Layer, Middleware & Dependencies (`app/api/`)

### `app/api/middleware.py`
- **What it is:** Security middleware.
- **Key Features:**
  - `RequestIDMiddleware`: Injects correlation ID (`X-Request-ID`).
  - `RateLimitBackend`: Pluggable architecture supporting `MemoryRateLimitBackend` (process-local) and `RedisRateLimitBackend` (distributed multi-worker).
  - `verify_api_key`: Enforces `X-API-Key` authentication when enabled using `SecretStr`.

### `app/api/dependencies.py`
- **What it is:** Dependency injection providers reusing shared service singletons from `app.state`.

### `app/api/ingest.py`
- **What it is:** `POST /api/v1/ingest` endpoint with chunked streaming body validation (preventing RAM exhaustion on oversized uploads), magic bytes check, auth, and safe error masking.

### `app/api/query.py`
- **What it is:** `POST /api/v1/query` endpoint with query length/whitespace validation, top_k bounds, rate limiting, auth, and grounded multi-modal answers.

### `app/api/documents.py`
- **What it is:** `GET /api/v1/documents` and `DELETE /api/v1/documents/{filename}` with complete filesystem and vector purging, with client-safe error masking and request ID tracking.

### `app/api/router.py`
- **What it is:** Aggregates all sub-routers under `/api/v1`.

---

## 9. Utilities & Infrastructure (`app/utils/`)

### `app/utils/file_utils.py`
- **What it is:** Security and filesystem utilities:
  - `validate_pdf_signature(content)`: `%PDF-` validation.
  - `sanitize_filename(name)`: Path traversal neutralization.
  - `generate_secure_upload_path(...)`: Unique collision-resistant paths.
  - `compute_file_hash(...)`: SHA-256 checksums.

### `app/utils/logging_config.py`
- **What it is:** Production logging configuration supporting both text and JSON formats.

---

## 10. Demonstration & Operational Scripts (`scripts/`)

### `scripts/seed_sample.py`
- **What it is:** Standalone end-to-end verification script:
  1. Generates a synthetic multi-page PDF with an embedded 500x300 RGB bar chart.
  2. Parses, chunks, crops images, generates dense VLM summaries, embeds vectors, and indexes into ChromaDB.
  3. Executes a trial semantic query and displays retrieved multi-modal citations.

---

## 11. Automated Test Suite (`tests/`)

The repository includes **37 automated tests** (100% pass rate):

| Test File | Tests | Coverage Scope |
|---|:---:|---|
| `tests/conftest.py` | Fixtures | Shared isolated test clients, temporary databases, and synthetic PDF builders. |
| `tests/test_api.py` | 6 tests | REST endpoints: root, health, liveness/readiness probes, file uploads, full RAG lifecycle. |
| `tests/test_security.py` | 8 tests | Path traversal, `%PDF-` magic bytes, spoofed uploads, blank PDFs, API key auth, rate limiting, and filesystem cleanup. |
| `tests/test_embedding_service.py` | 2 tests | Vector dimensionality (384) and batch embedding normalization. |
| `tests/test_ingestion.py` | 1 test | End-to-end ingestion pipeline and ChromaDB index verification. |
| `tests/test_pdf_parser.py` | 4 tests | PyMuPDF parsing, large figure extraction (>100px), icon filtering (<100px), and async offload. |
| `tests/test_text_chunker.py` | 5 tests | Parameter validation, whitespace handling, paragraph splitting, and sliding overlap. |
| `tests/test_vectorstore.py` | 5 tests | ChromaDB upsert, cosine distance retrieval, metadata filtering, and source deletion. |
| `tests/test_vlm_service.py` | 6 tests | File validation, synthetic fallback, batch summarization, mocked Gemini/OpenAI, and failure degradation. |

---

## 12. Data & Storage Directories (`data/`, `chroma_db/`)

- `data/uploads/`: Ingested PDF documents (tracked via `.gitkeep`).
- `data/extracted_images/`: Cropped high-resolution figures served via `/static/images` (tracked via `.gitkeep`).
- `chroma_db/`: On-disk persistent ChromaDB database containing HNSW vector indexes.
