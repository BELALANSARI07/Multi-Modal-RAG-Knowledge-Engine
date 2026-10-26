---
name: blast-readiness-audit
description: >-
  Audits production readiness using the B.L.A.S.T. protocol, runs static security
  and dependency sweeps, and autonomously generates resilient CI/CD pipelines
  with post-deployment health verification and automated rollback capabilities.
---

# B.L.A.S.T. Production Readiness Audit & Automated Rollback Guide

Use this skill to audit backend services and generate resilient CI/CD deployment pipelines featuring active post-deployment health validation and automated rollback capabilities.

## Protocol 0: Initialization & Architecture Baseline

1. **Create Architecture Source of Truth**:
   - Generate `Readiness_Audit_Map.md` in the repository root mapping:
     - Technology stack, runtime, frameworks, and database engines.
     - Ingestion, retrieval, and request flow diagrams.
     - Security boundaries, authentication points, and rate limiting backends.
     - Operational health probe endpoints and contracts.
2. **Confirm Deployment Parameters**:
   - Target deployment environment (AWS ECS/Fargate, GCP Cloud Run, Azure Container Apps, VPS/Docker).
   - Deployment branching strategy (e.g. direct push to `main`, semantic release tags `v*`).
   - Rollback mechanism (reverting to previous container image tag / task definition revision).
   - Verified environment variable baseline (`.env.example`).

## Phase 1: Blueprint

1. **Security & Auth Verification**:
   - Verify credentials use Pydantic `SecretStr` models to prevent unmasked leaks.
   - Audit CORS configuration to ensure origins are isolated and not using `*` with credentials.
   - Ensure generic client-safe error messages with `X-Request-ID` correlation.
2. **Performance & Latency Thresholds**:
   - Define strict latency bounds: Liveness < 100ms, Readiness < 1500ms, Smoke Query < 2000ms.
3. **Resilient Pipeline Staging**:
   - Stage 1: Quality & Pre-Deployment Audit (Ruff, Gitleaks, Pytest Matrix).
   - Stage 2: Container Build & Push (Multi-stage Dockerfile).
   - Stage 3: Deployment (ECS service update, recording prior stable revision).
   - Stage 4: Post-Deployment Verification (Liveness, readiness, smoke query).
   - Stage 5: Conditional Automated Rollback (`if: failure()`).

## Phase 2: Link (Diagnostic & Secret Tooling)

1. **Dependency Connectivity**:
   - Build `tools/check_dependencies.py` to test database read/write cycles, embedding inference, and storage permissions.
2. **Secret Mapping**:
   - Build `tools/verify_secrets.py` to audit required GitHub Secrets (`AWS_ACCESS_KEY_ID`, `ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE`, `PRODUCTION_URL`).

## Phase 3: Architect (Deterministic SOPs & Generators)

1. **Operational SOPs**:
   - Create `architecture/deployment_sop.md`: Explicit pre-deployment gates and failure conditions.
   - Create `architecture/rollback_matrix.md`: Decision matrix and runbook with target RTO < 45s.
2. **Audit Routing Engine**:
   - Build `tools/audit_engine.py`: Scans for hardcoded credentials, Dockerfile non-root user (`appuser`), and schema bounds.
3. **Health Gate & Rollback Handler**:
   - Build `tools/post_deploy_health_check.py`: Polls probes and enforces status == 200 and latency < 2.0s.
   - Build `tools/rollback_handler.py`: Reverts deployment to prior stable task definition and sends incident webhook.
4. **Autonomous CI/CD Generator**:
   - Build `tools/generate_pipeline.py` to output `.github/workflows/deploy.yml`.

## Phase 4: Stylize & Trigger

1. **Deliver Production Readiness Report**:
   - Output `Production_Readiness_Report.md` with Pass/Fail results and telemetry benchmarks.
2. **Validate Pipeline Delivery**:
   - Verify YAML syntax and ensure conditional rollback jobs are explicitly commented for review.
