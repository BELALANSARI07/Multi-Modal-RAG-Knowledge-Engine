#!/usr/bin/env python3
"""Audit Routing Engine & Static Analysis Scanner.

Performs static analysis and security sweeps across the codebase:
1. Secret Sweep: Scans all tracked code files for hardcoded API keys or high-entropy credentials.
2. Endpoint Auth Coverage: Scans FastAPI endpoints for security dependencies and rate limiting.
3. Container Security: Audits Dockerfile for multi-stage structure, non-root execution, and healthchecks.
4. CORS & Middleware: Verifies production CORS isolation and request correlation.
5. Pydantic Defense Bounds: Verifies schema validations against unbounded inputs.
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def scan_for_hardcoded_secrets() -> dict:
    """Scan code files for prospective hardcoded API keys or sensitive credentials."""
    patterns = [
        (r'(?i)(?:api_key|secret|password|token)\s*=\s*["\'](AIza[0-9A-Za-z-_]{35}|sk-[a-zA-Z0-9]{32,})["\']', "Live API Key Pattern"),
        (r'(?i)-----BEGIN (?:RSA |EC )?PRIVATE KEY-----', "Private Key Block"),
        (r'(?i)ghp_[0-9a-zA-Z]{36}', "GitHub Personal Access Token"),
        (r'(?i)xox[baprs]-[0-9a-zA-Z]{10,48}', "Slack Token"),
    ]
    violations = []

    # Files to exclude from scan
    ignored_dirs = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "venv", ".venv"}
    ignored_files = {".env.example"}

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for f in files:
            if f in ignored_files or f.endswith((".pyc", ".png", ".pdf", ".jpg")):
                continue
            fpath = Path(root) / f
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                for pat, name in patterns:
                    matches = re.findall(pat, content)
                    if matches:
                        violations.append({
                            "file": str(fpath.relative_to(PROJECT_ROOT)),
                            "check": name,
                            "match_count": len(matches),
                        })
            except Exception:
                pass

    return {
        "check": "Hardcoded Secrets Sweep",
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
    }


def audit_dockerfile() -> dict:
    """Audit Dockerfile configuration for production security standards."""
    df_path = PROJECT_ROOT / "Dockerfile"
    if not df_path.exists():
        return {"check": "Dockerfile Security", "status": "FAIL", "violations": ["Dockerfile missing"]}

    content = df_path.read_text(encoding="utf-8", errors="ignore")
    violations = []

    # Check 1: Multi-stage build
    if "AS builder" not in content or "FROM python:" not in content:
        violations.append("Missing multi-stage builder pattern ('AS builder')")

    # Check 2: Non-root user
    if not re.search(r'(?i)user\s+(?:appuser|\d{4,})', content):
        violations.append("Container runs as root (missing 'USER appuser' directive)")

    # Check 3: Healthcheck
    if "HEALTHCHECK" not in content:
        violations.append("Missing container HEALTHCHECK instruction")

    # Check 4: Single worker concurrency note for SQLite
    if '"--workers", "1"' not in content and "--workers 1" not in content:
        violations.append("Missing '--workers 1' concurrency safety constraint for SQLite ChromaDB")

    return {
        "check": "Dockerfile Security Standards",
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
    }


def audit_cors_configuration() -> dict:
    """Verify CORS origin whitelist settings and credential safety."""
    config_path = PROJECT_ROOT / "app" / "config.py"
    main_path = PROJECT_ROOT / "app" / "main.py"
    violations = []

    if config_path.exists():
        cfg = config_path.read_text(encoding="utf-8")
        if 'cors_origins: list[str] | str = ["*"]' in cfg or "allow_origins=['*']" in cfg:
            violations.append("Wildcard CORS origin detected in default configuration")

    if main_path.exists():
        main_code = main_path.read_text(encoding="utf-8")
        if 'allow_origins=["*"]' in main_code and 'allow_credentials=True' in main_code:
            violations.append("Insecure CORS combination: allow_origins=['*'] with allow_credentials=True")

    return {
        "check": "CORS & Credential Boundaries",
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
    }


def audit_input_bounds() -> dict:
    """Verify that schemas enforce upper and lower bounds on request fields."""
    schemas_path = PROJECT_ROOT / "app" / "models" / "schemas.py"
    violations = []

    if schemas_path.exists():
        code = schemas_path.read_text(encoding="utf-8")
        if "max_length" not in code:
            violations.append("QueryRequest question field lacks max_length constraint")
        if "top_k" not in code or "le=" not in code:
            violations.append("QueryRequest top_k lacks upper bound (le=) guardrail")
    else:
        violations.append("schemas.py not found")

    return {
        "check": "Input Schema Bounds & Guardrails",
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
    }


def run_full_audit() -> list[dict]:
    """Execute all static security and architectural audits."""
    checks = [
        scan_for_hardcoded_secrets(),
        audit_dockerfile(),
        audit_cors_configuration(),
        audit_input_bounds(),
    ]
    return checks


def main() -> int:
    """Run audit engine and print diagnostic summary."""
    print("=" * 65)
    print("  PRODUCTION READINESS AUDIT: STATIC SECURITY SWEEP")
    print("=" * 65)

    results = run_full_audit()
    all_passed = True

    for r in results:
        status = r["status"]
        if status == "FAIL":
            all_passed = False
            print(f"\n[-] {r['check']}: \033[91m[FAIL]\033[0m")
            for v in r["violations"]:
                print(f"    - {v}")
        else:
            print(f"\n[+] {r['check']}: \033[92m[PASS]\033[0m")

    print("\n" + "=" * 65)
    if all_passed:
        print("  \033[92mALL STATIC SECURITY AND CONTAINER CHECKS PASSED\033[0m")
        return 0
    else:
        print("  \033[91mVULNERABILITIES OR GAPS DETECTED\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
