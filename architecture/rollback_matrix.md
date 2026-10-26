# Automated Rollback Decision Matrix & Runbook
**Service: Multi-Modal RAG Knowledge Engine**
**Infrastructure: AWS ECS Fargate & Application Load Balancer**

---

## 1. Decision Matrix & Action Triggers

| Trigger Condition | Detection Point | Automated Action | Target RTO | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **Liveness Failure (HTTP != 200)** | Post-deploy probe `GET /health/live` | Immediate service revert to previous Task Def | < 45s | P0 |
| **Readiness Failure (DB offline)** | Post-deploy probe `GET /health/ready` | Revert to previous Task Def & purge cache | < 45s | P0 |
| **Latency Breach (> 2000ms)** | Smoke query `POST /api/v1/query` | Automated rollback to prior verified image | < 60s | P1 |
| **Container Crash / OOM (137)** | AWS ECS Service events & CloudWatch | Revert to previous Task Def; rollback alert | < 60s | P0 |
| **Missing Environment Secrets** | Gating tool `tools/verify_secrets.py` | Pipeline fail-fast before image build | 0s (no deploy) | P2 |
| **SQLite Concurrency Lock** | Ingestion lock detected in logs | Revert deployment & verify `--workers 1` | < 60s | P1 |

---

## 2. Automated Rollback Runbook (AWS ECS)

### Phase 1: Detection
GitHub Actions post-deploy job executes `tools/post_deploy_health_check.py`.
If any test returns non-zero, the job terminates with exit code 1.

### Phase 2: Execution (via `tools/rollback_handler.py`)
```bash
# 1. Fetch current and previous task definitions
CURRENT_TASK_DEF=$(aws ecs describe-services --cluster "${ECS_CLUSTER}" --services "${ECS_SERVICE}" --query "services[0].taskDefinition" --output text)
PREVIOUS_STABLE_TASK_DEF=$(cat .previous_stable_task_def || echo "")

# 2. Revert service to previous stable task definition
aws ecs update-service \
    --cluster "${ECS_CLUSTER}" \
    --service "${ECS_SERVICE}" \
    --task-definition "${PREVIOUS_STABLE_TASK_DEF}" \
    --force-new-deployment

# 3. Wait until the reverted service reaches steady state
aws ecs wait services-stable \
    --cluster "${ECS_CLUSTER}" \
    --services "${ECS_SERVICE}"
```

### Phase 3: Incident Notification
The rollback handler transmits a JSON payload to `ROLLBACK_WEBHOOK` (Slack / Teams / PagerDuty):
```json
{
  "status": "ROLLBACK_COMPLETED",
  "service": "multimodal-rag-engine",
  "reverted_from_sha": "${{ github.sha }}",
  "restored_task_def": "${PREVIOUS_STABLE_TASK_DEF}",
  "trigger": "Post-deploy health probe failed",
  "action_required": "Inspect CloudWatch logs and pytest artifacts"
}
```

---

## 3. Manual Override & Emergency Recovery

If the automated rollback script cannot communicate with the AWS API:
1. Open AWS ECS Console -> Clusters -> `${ECS_CLUSTER}` -> Services -> `${ECS_SERVICE}`.
2. Select **Update Service**.
3. In **Revision**, select the previous stable Task Definition revision number (e.g. `multimodal-rag-task:N-1`).
4. Select **Force new deployment** and save.
5. Verify health: `curl -f https://<ALB_DNS>/health/ready`.
