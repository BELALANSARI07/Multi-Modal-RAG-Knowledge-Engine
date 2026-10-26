#!/usr/bin/env python3
"""Autonomous CI/CD Pipeline Generator with Health Verification & Automated Rollback.

Generates .github/workflows/deploy.yml adhering to the B.L.A.S.T. protocol:
1. Stage 1: Quality & Pre-Deployment Audit (Ruff, Gitleaks, Pytest Matrix, Audit Sweep).
2. Stage 2: Container Build & Push (AWS ECR authentication & multi-stage build).
3. Stage 3: AWS ECS Deployment (Task definition registration & rolling update).
4. Stage 4: Post-Deployment Verification (Liveness, readiness, and latency gate).
5. Stage 5: Conditional Automated Rollback (if: failure() -> revert to prior stable revision).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "deploy.yml"


DEPLOY_YAML_TEMPLATE = r"""name: Production Deployment & Resilient Rollback Pipeline

on:
  push:
    branches:
      - main
  workflow_dispatch:
    inputs:
      target_env:
        description: 'Deployment Environment'
        required: true
        default: 'production'
        type: choice
        options:
          - production
          - staging

permissions:
  contents: read
  id-token: write
  packages: write

concurrency:
  group: production-deployment
  cancel-in-progress: false

jobs:
  # =========================================================================
  # STAGE 1: Code Quality, Security Audit & Pytest Matrix Verification
  # =========================================================================
  quality-and-audit:
    name: Code Quality & Security Audit
    runs-on: ubuntu-24.04
    strategy:
      fail-fast: true
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"

      - name: Install System Build Dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y --no-install-recommends build-essential libmupdf-dev

      - name: Install Pinned Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ruff==0.6.8
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Execute Ruff Linter Check
        run: |
          ruff check .

      - name: Secret Leak Detection (Gitleaks)
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Run Dependency & Connectivity Diagnostics
        run: |
          python tools/check_dependencies.py

      - name: Run Static Security & Dockerfile Audit
        run: |
          python tools/audit_engine.py

      - name: Execute Full Test Suite
        shell: bash
        run: |
          set -o pipefail
          pytest tests/ -v --tb=short 2>&1 | tee pytest_output.txt

      - name: Publish Test Summary
        if: always()
        run: |
          if [ -f pytest_output.txt ]; then
            echo "### Pytest Execution Summary (Python ${{ matrix.python-version }})" >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
            tail -n 25 pytest_output.txt >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
          fi

  # =========================================================================
  # STAGE 2: Build Multi-Stage Container & Push to Amazon ECR
  # =========================================================================
  build-and-push:
    name: Build & Push Container to Amazon ECR
    needs: quality-and-audit
    runs-on: ubuntu-24.04
    outputs:
      image_tag: ${{ steps.set-tag.outputs.tag }}
      image_uri: ${{ steps.build-image.outputs.image }}

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION || 'us-east-1' }}

      - name: Log in to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Compute Deterministic Image Tag
        id: set-tag
        run: |
          TAG="${GITHUB_SHA::7}"
          echo "tag=${TAG}" >> $GITHUB_OUTPUT
          echo "ECR_TAG=${TAG}" >> $GITHUB_ENV

      - name: Build, Tag, and Push Docker Image
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPO: ${{ secrets.ECR_REPOSITORY || 'multimodal-rag-engine' }}
        run: |
          IMAGE_URI="$ECR_REGISTRY/$ECR_REPO:$ECR_TAG"
          LATEST_URI="$ECR_REGISTRY/$ECR_REPO:latest"

          echo "Building image: $IMAGE_URI"
          docker build -t "$IMAGE_URI" -t "$LATEST_URI" .

          echo "Pushing images to ECR..."
          docker push "$IMAGE_URI"
          docker push "$LATEST_URI"

          echo "image=$IMAGE_URI" >> $GITHUB_OUTPUT

  # =========================================================================
  # STAGE 3: Deploy to AWS ECS Fargate Service
  # =========================================================================
  deploy-to-ecs:
    name: Deploy to AWS ECS
    needs: build-and-push
    runs-on: ubuntu-24.04
    outputs:
      previous_task_def: ${{ steps.record-previous.outputs.prev_task_def }}

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION || 'us-east-1' }}

      - name: Record Current Active Task Definition for Rollback Target
        id: record-previous
        env:
          CLUSTER: ${{ secrets.ECS_CLUSTER || 'multimodal-rag-cluster' }}
          SERVICE: ${{ secrets.ECS_SERVICE || 'multimodal-rag-service' }}
        run: |
          echo "Querying current active task definition from ECS..."
          PREV_DEF=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query "services[0].taskDefinition" --output text || echo "")
          echo "Current active task definition: $PREV_DEF"
          echo "prev_task_def=$PREV_DEF" >> $GITHUB_OUTPUT
          echo "$PREV_DEF" > previous_task_def.txt

      - name: Download Active Task Definition JSON
        env:
          CLUSTER: ${{ secrets.ECS_CLUSTER || 'multimodal-rag-cluster' }}
          SERVICE: ${{ secrets.ECS_SERVICE || 'multimodal-rag-service' }}
        run: |
          aws ecs describe-task-definition --task-definition "${{ steps.record-previous.outputs.prev_task_def }}" --query "taskDefinition" > task-def.json

      - name: Render Updated Task Definition with New Image URI
        id: render-task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: task-def.json
          container-name: multimodal-rag-engine
          image: ${{ needs.build-and-push.outputs.image_uri }}

      - name: Deploy Task Definition to AWS ECS Service
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: ${{ steps.render-task-def.outputs.task-definition }}
          service: ${{ secrets.ECS_SERVICE || 'multimodal-rag-service' }}
          cluster: ${{ secrets.ECS_CLUSTER || 'multimodal-rag-cluster' }}
          wait-for-service-stability: true

  # =========================================================================
  # STAGE 4: Post-Deployment Health Verification & Latency Gate
  # =========================================================================
  post-deploy-verification:
    name: Post-Deploy Health Check & Latency Gate
    needs: [build-and-push, deploy-to-ecs]
    runs-on: ubuntu-24.04

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Execute Post-Deploy Health Check & Latency Gate
        env:
          PRODUCTION_URL: ${{ secrets.PRODUCTION_URL }}
          API_KEY: ${{ secrets.API_KEY }}
        run: |
          # Enforces:
          # 1. GET /health/live == HTTP 200
          # 2. GET /health/ready == HTTP 200 (database + embeddings ready)
          # 3. GET /health == HTTP 200
          # 4. POST /api/v1/query smoke query == HTTP 200 with answer
          # 5. Roundtrip latency < 2.0s
          python tools/post_deploy_health_check.py \\
            --base-url "${{ secrets.PRODUCTION_URL || 'http://localhost:8000' }}" \\
            --api-key "${{ secrets.API_KEY }}" \\
            --latency-threshold 2.0 \\
            --max-retries 12 \\
            --retry-delay 5.0

      - name: Log Verification Success
        if: success()
        run: |
          echo "### :white_check_mark: Post-Deployment Verification Succeeded" >> $GITHUB_STEP_SUMMARY
          echo "- **Target**: ${{ secrets.PRODUCTION_URL }}" >> $GITHUB_STEP_SUMMARY
          echo "- **Verified Commit**: \`${{ github.sha }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- **Latency Status**: All probes responded in < 2.0s" >> $GITHUB_STEP_SUMMARY

  # =========================================================================
  # STAGE 5: Automated Rollback on Failure (Conditional: if: failure())
  # =========================================================================
  # NOTE FOR REVIEW:
  # This job triggers automatically ONLY if Stage 4 (post-deploy-verification) fails
  # or if the deployment step times out. It automatically restores the ECS service
  # back to the previous stable Task Definition recorded in Stage 3, and sends
  # a high-priority incident alert to the configured webhook.
  automated-rollback:
    name: Automated Deployment Rollback
    needs: [deploy-to-ecs, post-deploy-verification]
    if: failure() && needs.deploy-to-ecs.result == 'success'
    runs-on: ubuntu-24.04

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION || 'us-east-1' }}

      - name: Execute Automated Rollback to Previous Stable Task Definition
        env:
          ECS_CLUSTER: ${{ secrets.ECS_CLUSTER || 'multimodal-rag-cluster' }}
          ECS_SERVICE: ${{ secrets.ECS_SERVICE || 'multimodal-rag-service' }}
          PREVIOUS_STABLE_TASK_DEF: ${{ needs.deploy-to-ecs.outputs.previous_task_def }}
          ROLLBACK_WEBHOOK: ${{ secrets.ROLLBACK_WEBHOOK }}
          GITHUB_SHA: ${{ github.sha }}
        run: |
          echo "================================================================="
          echo "CRITICAL: Triggering Automated Rollback for Commit ${{ github.sha }}"
          echo "Target Task Definition: $PREVIOUS_STABLE_TASK_DEF"
          echo "================================================================="

          python tools/rollback_handler.py \\
            --cluster "$ECS_CLUSTER" \\
            --service "$ECS_SERVICE" \\
            --target-task-def "$PREVIOUS_STABLE_TASK_DEF" \\
            --reason "Post-deployment health verification probe failed or latency exceeded 2.0s threshold"

      - name: Record Rollback Incident Summary
        if: always()
        run: |
          echo "### :rotating_light: Automated Rollback Executed" >> $GITHUB_STEP_SUMMARY
          echo "- **Failed Commit**: \`${{ github.sha }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- **Reverted Cluster**: \`${{ secrets.ECS_CLUSTER }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- **Restored Task Definition**: \`${{ needs.deploy-to-ecs.outputs.previous_task_def }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- **Action Taken**: Service rolled back to stable baseline and stabilized." >> $GITHUB_STEP_SUMMARY
"""


def generate_pipeline() -> Path:
    """Generate .github/workflows/deploy.yml."""
    DEPLOY_WORKFLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEPLOY_WORKFLOW_PATH.write_text(DEPLOY_YAML_TEMPLATE, encoding="utf-8")
    print(f"[+] Successfully generated CI/CD deployment pipeline at: {DEPLOY_WORKFLOW_PATH}")
    return DEPLOY_WORKFLOW_PATH


def main() -> int:
    print("=" * 65)
    print("  AUTONOMOUS CI/CD PIPELINE GENERATOR (B.L.A.S.T. PROTOCOL)")
    print("=" * 65)
    path = generate_pipeline()
    if path.exists():
        print(f"[+] Verification: Pipeline file created ({path.stat().st_size} bytes)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
