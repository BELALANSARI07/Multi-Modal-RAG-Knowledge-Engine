#!/usr/bin/env python3
"""Production Startup Script for Multi-Modal RAG Knowledge Engine.

Pre-flight checks:
1. Environment file (.env) verification.
2. Persistent directory initialization.
3. Single-worker concurrency enforcement for SQLite ChromaDB safety.
4. Starts Uvicorn production server.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def preflight_setup() -> None:
    """Ensure environment and persistence directories are properly initialized."""
    print("=" * 65)
    print("  MULTI-MODAL RAG ENGINE — PRODUCTION STARTUP SEQUENCE")
    print("=" * 65)

    # 1. Check .env file
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"
    if not env_file.exists():
        if env_example.exists():
            print("[+] .env file not found. Initializing from .env.example...")
            shutil.copy(env_example, env_file)
        else:
            print("[-] Warning: Neither .env nor .env.example was found.")

    # 2. Ensure persistence directories
    dirs = [
        PROJECT_ROOT / "data" / "uploads",
        PROJECT_ROOT / "data" / "extracted_images",
        PROJECT_ROOT / "chroma_db",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("[+] Persistence directories verified.")

    # 3. Informational message
    print("[+] Concurrency: Locked to 1 Uvicorn worker for SQLite ChromaDB thread safety.")
    print("[+] Health probes: http://localhost:8000/health/live and /health/ready")
    print("=" * 65)


def start_server() -> None:
    """Launch Uvicorn with production flags."""
    preflight_setup()
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        os.getenv("HOST", "0.0.0.0"),
        "--port",
        os.getenv("PORT", "8000"),
        "--workers",
        "1",
        "--proxy-headers",
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[+] Service stopped gracefully by operator.")


if __name__ == "__main__":
    start_server()
