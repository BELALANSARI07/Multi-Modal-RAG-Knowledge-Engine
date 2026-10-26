#!/usr/bin/env python3
"""Post-Deployment Health Verification & Latency Gate Tool.

Pings critical endpoints of the deployed production service:
1. Liveness Probe: GET /health/live
2. Readiness Probe: GET /health/ready (validates ChromaDB & embedding engine)
3. Service Telemetry: GET /health
4. Smoke Query Probe: POST /api/v1/query (validates end-to-end RAG pipeline)

Enforces strict latency thresholds (< 2.0s) and HTTP 200 status codes.
Exits with code 0 on success; exits with code 1 on failure to trigger CI automated rollback.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def make_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any] | str, float]:
    """Execute HTTP request and return (status_code, body, elapsed_seconds)."""
    headers = headers or {}
    req_data = None
    if data is not None:
        req_data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - start
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = body
            return resp.status, parsed, elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - start
        body = e.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body
        return e.code, parsed, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        return 0, str(e), elapsed


def poll_endpoint_until_healthy(
    url: str,
    max_retries: int,
    retry_delay: float,
    expected_status: int = 200,
    headers: dict[str, str] | None = None,
) -> tuple[bool, int, Any, float]:
    """Poll an endpoint until it returns the expected status code or exhausts retries."""
    for attempt in range(1, max_retries + 1):
        status, body, elapsed = make_request(url, method="GET", headers=headers, timeout=10.0)
        print(f"    [Attempt {attempt}/{max_retries}] {url} -> HTTP {status} ({elapsed:.3f}s)")
        if status == expected_status:
            return True, status, body, elapsed
        time.sleep(retry_delay)
    return False, status, body, elapsed


def run_health_checks(
    base_url: str,
    api_key: str | None = None,
    max_retries: int = 12,
    retry_delay: float = 5.0,
    latency_threshold: float = 2.0,
) -> bool:
    """Execute the full battery of post-deployment verification checks."""
    base = base_url.rstrip("/")
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    print("=" * 70)
    print("  POST-DEPLOYMENT HEALTH & READINESS VERIFICATION")
    print(f"  Target: {base}")
    print(f"  Latency Threshold: < {latency_threshold:.2f}s | Max Polling Retries: {max_retries}")
    print("=" * 70)

    # 1. Liveness Probe (Poll until container is responsive)
    print("\n[+] 1. Checking Liveness Probe (/health/live)...")
    live_url = f"{base}/health/live"
    ok, status, body, elapsed = poll_endpoint_until_healthy(
        live_url, max_retries=max_retries, retry_delay=retry_delay, headers=headers
    )
    if not ok or status != 200:
        print(f"    [-] Liveness probe failed. Target did not respond with HTTP 200. Body: {body}")
        return False
    if elapsed > latency_threshold:
        print(f"    [-] Liveness probe exceeded latency threshold ({elapsed:.2f}s > {latency_threshold:.2f}s)")
        return False
    print(f"    --> [PASS] Liveness OK ({elapsed:.3f}s)")

    # 2. Readiness Probe (ChromaDB + Embedding Engine initialized)
    print("\n[+] 2. Checking Readiness Probe (/health/ready)...")
    ready_url = f"{base}/health/ready"
    status, body, elapsed = make_request(ready_url, method="GET", headers=headers, timeout=10.0)
    print(f"    {ready_url} -> HTTP {status} ({elapsed:.3f}s)")
    if status != 200:
        print(f"    [-] Readiness probe failed: HTTP {status}. Body: {body}")
        return False
    if isinstance(body, dict) and body.get("status") != "ready":
        print(f"    [-] Readiness probe returned non-ready payload: {body}")
        return False
    if elapsed > latency_threshold:
        print(f"    [-] Readiness latency breach ({elapsed:.2f}s > {latency_threshold:.2f}s)")
        return False
    print(f"    --> [PASS] Readiness OK ({elapsed:.3f}s)")

    # 3. System Telemetry Probe (/health)
    print("\n[+] 3. Checking System Telemetry (/health)...")
    health_url = f"{base}/health"
    status, body, elapsed = make_request(health_url, method="GET", headers=headers, timeout=10.0)
    print(f"    {health_url} -> HTTP {status} ({elapsed:.3f}s)")
    if status != 200:
        print(f"    [-] Telemetry probe failed: HTTP {status}")
        return False
    print(f"    --> [PASS] Telemetry OK ({elapsed:.3f}s)")

    # 4. Smoke Query Verification Probe (/api/v1/query)
    print("\n[+] 4. Executing Smoke Query Probe (/api/v1/query)...")
    query_url = f"{base}/api/v1/query"
    smoke_payload = {
        "question": "What is the system operational readiness status?",
        "top_k": 1,
        "include_sources": True,
    }
    status, body, elapsed = make_request(
        query_url, method="POST", headers=headers, data=smoke_payload, timeout=30.0
    )
    print(f"    {query_url} -> HTTP {status} ({elapsed:.3f}s)")
    if status != 200:
        print(f"    [-] Smoke query failed with HTTP {status}. Body: {body}")
        return False
    if not isinstance(body, dict) or "answer" not in body:
        print(f"    [-] Smoke query response missing 'answer' field: {body}")
        return False
    print(f"    --> [PASS] Smoke Query Succeeded ({elapsed:.3f}s)")

    print("\n" + "=" * 70)
    print("  \033[92mALL POST-DEPLOYMENT HEALTH CHECKS PASSED SUCCESSFULLY\033[0m")
    print("=" * 70)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-Deployment Health & Latency Verifier")
    parser.add_argument("--base-url", default=os.getenv("PRODUCTION_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", None))
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--latency-threshold", type=float, default=2.0)
    args = parser.parse_args()

    success = run_health_checks(
        base_url=args.base_url,
        api_key=args.api_key,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        latency_threshold=args.latency_threshold,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
