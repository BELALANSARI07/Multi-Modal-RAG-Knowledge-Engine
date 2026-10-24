# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-24

### Added
- **Security Hardening:** PDF magic bytes (`%PDF-`) verification and path-traversal-proof filename sanitization.
- **Authentication & Authorization:** Header-based API key authentication (`X-API-Key`).
- **Rate Limiting:** Token-bucket sliding window rate limiting on `/ingest` and `/query` endpoints.
- **Containerization:** Production multi-stage `Dockerfile` and `docker-compose.yml` with persistent volume mappings and container health check.
- **CI/CD Pipeline:** GitHub Actions workflow (`.github/workflows/ci.yml`) testing Python 3.10, 3.11, and 3.12 with Ruff linting.
- **Observability:** `X-Request-ID` correlation middleware and `/health/live` & `/health/ready` probe endpoints.
- **Transactional Ingestion:** Automated rollback and cleanup of disk files if embedding or ChromaDB insertion encounters errors.
- **Document Management:** Cascade filesystem deletion removing source PDF and extracted images when document is deleted.
- **Testing:** Comprehensive test suite expansion with `tests/conftest.py` shared fixtures and `tests/test_security.py`.
- **Project Packaging:** Modern `pyproject.toml` and `.python-version` specifications.

## [1.0.0] - 2026-10-24

### Added
- Initial release of the Multi-Modal RAG Knowledge Engine.
- Multi-modal PDF ingestion via PyMuPDF (fitz) and Pillow.
- Recursive character text chunker preserving paragraph and sentence boundaries.
- Semantic image processing via Gemini 1.5 Flash and GPT-4o-mini Vision APIs.
- Unified vectorization into ChromaDB using local sentence-transformers with ONNX Runtime fallback.
- FastAPI endpoints for ingestion, query, and document management.
