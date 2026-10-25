# ============================================================
# Multi-Modal RAG Knowledge Engine — Multi-Stage Production Dockerfile
# ============================================================

# ------------------------------------------------------------
# Stage 1: Build & Wheel Compilation
# ------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Pre-compile binary wheels to eliminate build tools in the final runner image
RUN pip install --upgrade pip && \
    pip wheel --wheel-dir=/install/wheels -r requirements.txt

# ------------------------------------------------------------
# Stage 2: Final Minimal Runtime Image
# ------------------------------------------------------------
FROM python:3.10-slim-bookworm AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install lightweight runtime dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pre-built wheels from builder stage
COPY --from=builder /install/wheels /wheels
RUN pip install --upgrade pip && \
    pip install /wheels/* && \
    rm -rf /wheels

# Create dedicated non-root application user for defense-in-depth
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/uploads /app/data/extracted_images /app/chroma_db && \
    chown -R appuser:appuser /app

# Copy application code with non-root ownership
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser README.md pyproject.toml ./

USER appuser

EXPOSE 8000

# Container liveness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Start Uvicorn: 1 worker is recommended for in-process SQLite ChromaDB to prevent write lock contention
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
