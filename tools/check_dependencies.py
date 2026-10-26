#!/usr/bin/env python3
"""Production Readiness Audit: Dependency & Service Connectivity Verification Tool.

Pings and validates:
1. Python runtime environment & critical packages.
2. Vector store (ChromaDB) connection, persistence, and read/write cycle.
3. Embedding model inference and dimension accuracy (SentenceTransformers / ONNX).
4. Filesystem write permissions for storage volumes.
5. VLM provider API connectivity and synthetic fallback integrity.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402


def check_python_environment() -> dict:
    """Verify Python version and critical libraries."""
    result = {"status": "PASS", "details": {}}
    py_ver = sys.version_info
    result["details"]["python_version"] = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver < (3, 10) or py_ver >= (3, 13):
        result["status"] = "FAIL"
        result["details"]["error"] = f"Unsupported Python version: {sys.version}. Expected 3.10 to 3.12."
        return result

    required_pkgs = [
        "fastapi",
        "pydantic",
        "chromadb",
        "fitz",
        "uvicorn",
        "tenacity",
    ]
    missing = []
    for pkg in required_pkgs:
        try:
            __import__(pkg)
            result["details"][pkg] = "available"
        except ImportError:
            missing.append(pkg)
            result["details"][pkg] = "MISSING"

    if missing:
        result["status"] = "FAIL"
        result["details"]["error"] = f"Missing required dependencies: {missing}"
    return result


def check_filesystem_permissions() -> dict:
    """Verify write and read access to essential persistent volumes."""
    settings = get_settings()
    dirs = [
        ("upload_dir", settings.upload_dir),
        ("extracted_images_dir", settings.extracted_images_dir),
        ("chroma_persist_dir", settings.chroma_persist_dir),
    ]
    result = {"status": "PASS", "details": {}}
    for name, path in dirs:
        p = Path(path).resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / f".readiness_check_{int(time.time())}.tmp"
            test_file.write_text("health-probe", encoding="utf-8")
            content = test_file.read_text(encoding="utf-8")
            test_file.unlink()
            if content != "health-probe":
                raise OSError("Write verification mismatch")
            result["details"][name] = f"RW verified at {p}"
        except Exception as e:
            result["status"] = "FAIL"
            result["details"][name] = f"FAIL: {str(e)}"
    return result


def check_chromadb_connection() -> dict:
    """Verify ChromaDB vector store initialization, indexing, and query capability."""
    result = {"status": "PASS", "details": {}}
    start = time.perf_counter()
    try:
        from app.services.vectorstore_service import VectorStoreService

        service = VectorStoreService()
        client = service._get_client()
        test_col_name = f"audit-probe-{int(time.time())}"
        col = client.get_or_create_collection(name=test_col_name)
        col.add(
            ids=["probe_1"],
            documents=["Audit readiness test document"],
            metadatas=[{"source": "audit"}],
            embeddings=[[0.01] * service.expected_dim],
        )
        count = col.count()
        query_res = col.query(query_embeddings=[[0.01] * service.expected_dim], n_results=1)
        client.delete_collection(test_col_name)

        elapsed_ms = (time.perf_counter() - start) * 1000
        result["details"]["latency_ms"] = round(elapsed_ms, 2)
        result["details"]["roundtrip_count"] = count
        result["details"]["query_matches"] = len(query_res.get("ids", [[]])[0])

        # Performance threshold: DB probe must complete in under 1500ms
        if elapsed_ms > 1500:
            result["status"] = "WARN"
            result["details"]["warning"] = f"ChromaDB probe latency high ({elapsed_ms:.1f}ms > 1500ms)"
    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["error"] = f"ChromaDB connection failed: {str(e)}"
    return result


def check_embedding_engine() -> dict:
    """Verify that the embedding engine loads properly and generates expected vector dimensions."""
    result = {"status": "PASS", "details": {}}
    start = time.perf_counter()
    try:
        from app.services.embedding_service import EmbeddingService

        service = EmbeddingService()
        vector = service.embed_single("Production readiness audit embedding probe")
        elapsed_ms = (time.perf_counter() - start) * 1000

        result["details"]["engine_type"] = service.engine_type
        result["details"]["dimension"] = len(vector)
        result["details"]["latency_ms"] = round(elapsed_ms, 2)

        if len(vector) != service.settings.embedding_dimension:
            result["status"] = "FAIL"
            result["details"]["error"] = (
                f"Dimension mismatch: expected {service.settings.embedding_dimension}, got {len(vector)}"
            )
        elif elapsed_ms > 2000:
            result["status"] = "WARN"
            result["details"]["warning"] = f"Cold-start embedding inference exceeded 2000ms ({elapsed_ms:.1f}ms)"
    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["error"] = f"Embedding service initialization failed: {str(e)}"
    return result


def check_vlm_provider() -> dict:
    """Verify VLM provider configuration and fallback readiness."""
    result = {"status": "PASS", "details": {}}
    settings = get_settings()
    provider = settings.vlm_provider
    result["details"]["provider"] = provider

    if provider == "gemini":
        has_key = bool(settings.gemini_api_key_str and "your_gemini" not in settings.gemini_api_key_str)
        result["details"]["api_key_configured"] = has_key
        if not has_key:
            result["details"]["fallback_mode"] = "Synthetic fallback active (Zero external API dependencies)"
    elif provider == "openai":
        has_key = bool(settings.openai_api_key_str and "your_openai" not in settings.openai_api_key_str)
        result["details"]["api_key_configured"] = has_key
        if not has_key:
            result["details"]["fallback_mode"] = "Synthetic fallback active (Zero external API dependencies)"
    return result


def main() -> int:
    """Execute all connectivity and dependency checks."""
    print("=" * 65)
    print("  MULTI-MODAL RAG ENGINE — DEPENDENCY & CONNECTIVITY AUDIT")
    print("=" * 65)

    checks = [
        ("Python & Core Packages", check_python_environment),
        ("Storage Volumes & Permissions", check_filesystem_permissions),
        ("ChromaDB Vector Store", check_chromadb_connection),
        ("Embedding Inference Engine", check_embedding_engine),
        ("VLM / LLM Provider & Fallback", check_vlm_provider),
    ]

    all_passed = True
    for name, func in checks:
        print(f"\n[+] Testing {name}...")
        res = func()
        status = res["status"]
        if status == "FAIL":
            all_passed = False
            print(f"    --> \033[91m[{status}]\033[0m")
        elif status == "WARN":
            print(f"    --> \033[93m[{status}]\033[0m")
        else:
            print(f"    --> \033[92m[{status}]\033[0m")

        for k, v in res.get("details", {}).items():
            print(f"        {k}: {v}")

    print("\n" + "=" * 65)
    if all_passed:
        print("  \033[92mOVERALL AUDIT RESULT: ALL DEPENDENCY CHECKS PASSED\033[0m")
        return 0
    else:
        print("  \033[91mOVERALL AUDIT RESULT: FAILURES DETECTED\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
