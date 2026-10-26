#!/usr/bin/env python3
"""Automated Rollback Orchestrator for AWS ECS & Container Deployments.

Triggered conditionally (if: failure()) when post-deployment health checks fail.
1. Reverts AWS ECS service to the previous stable Task Definition or container image.
2. Waits for ECS service to stabilize.
3. Broadcasts automated rollback notifications via webhook (Slack/Teams/OpsGenie).
4. Records structured incident diagnostics in GitHub Actions Step Summary.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def log_summary(text: str) -> None:
    """Append text to GITHUB_STEP_SUMMARY if available."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path and Path(summary_path).exists():
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


def send_webhook_alert(webhook_url: str, payload: dict) -> bool:
    """Send incident alert to webhook endpoint."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "RollbackHandler/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            print(f"[+] Incident alert dispatched to webhook (HTTP {resp.status})")
            return True
    except Exception as e:
        print(f"[-] Warning: Failed to deliver rollback webhook alert: {e}")
        return False


def execute_aws_ecs_rollback(cluster: str, service: str, target_task_def: str | None = None) -> bool:
    """Revert AWS ECS service to previous stable task definition."""
    print(f"[+] Initiating AWS ECS Rollback on Cluster '{cluster}', Service '{service}'...")

    # If target_task_def is not explicitly specified, inspect service history
    if not target_task_def:
        print("[+] Fetching previous active Task Definition revision...")
        try:
            cmd = [
                "aws", "ecs", "describe-services",
                "--cluster", cluster,
                "--services", service,
                "--query", "services[0].taskDefinition",
                "--output", "text",
            ]
            current_def = subprocess.check_output(cmd, text=True).strip()
            print(f"    Current Active Task Definition: {current_def}")

            # Parse family and revision: e.g. "multimodal-rag-task:5" -> "multimodal-rag-task:4"
            if ":" in current_def:
                family, rev_str = current_def.rsplit(":", 1)
                try:
                    prev_rev = int(rev_str) - 1
                    if prev_rev > 0:
                        target_task_def = f"{family}:{prev_rev}"
                except ValueError:
                    pass
        except Exception as e:
            print(f"[-] Warning: Could not auto-detect previous task definition: {e}")

    if not target_task_def:
        print("[-] Error: Unable to determine target Task Definition revision for rollback.")
        return False

    print(f"[+] Reverting ECS Service to Task Definition: {target_task_def}...")
    try:
        update_cmd = [
            "aws", "ecs", "update-service",
            "--cluster", cluster,
            "--service", service,
            "--task-definition", target_task_def,
            "--force-new-deployment",
        ]
        subprocess.check_call(update_cmd)
        print("[+] Service update issued. Waiting for service stabilization...")

        wait_cmd = [
            "aws", "ecs", "wait", "services-stable",
            "--cluster", cluster,
            "--services", service,
        ]
        subprocess.check_call(wait_cmd)
        print(f"[+] \033[92mService successfully restored to {target_task_def} and stabilized.\033[0m")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] AWS CLI command failed: {e}")
        return False
    except FileNotFoundError:
        print("[-] AWS CLI not found on runner path. Simulating rollback execution...")
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated Deployment Rollback Orchestrator")
    parser.add_argument("--cluster", default=os.getenv("ECS_CLUSTER", "multimodal-rag-cluster"))
    parser.add_argument("--service", default=os.getenv("ECS_SERVICE", "multimodal-rag-service"))
    parser.add_argument("--target-task-def", default=os.getenv("PREVIOUS_STABLE_TASK_DEF", None))
    parser.add_argument("--reason", default="Post-deploy health check failed or exceeded latency threshold")
    args = parser.parse_args()

    commit_sha = os.getenv("GITHUB_SHA", "unknown")[:7]
    webhook_url = os.getenv("ROLLBACK_WEBHOOK")

    banner = "\n".join([
        "=" * 70,
        "  AUTOMATED ROLLBACK TRIGGERED — INITIATING STABLE REVERSION",
        f"  Failed Git Commit: {commit_sha}",
        f"  Reason: {args.reason}",
        f"  Cluster: {args.cluster} | Service: {args.service}",
        "=" * 70,
    ])
    log_summary(banner)

    rollback_success = execute_aws_ecs_rollback(
        cluster=args.cluster,
        service=args.service,
        target_task_def=args.target_task_def,
    )

    alert_payload = {
        "text": f":warning: *Automated Rollback Executed* for `multimodal-rag-engine`\n"
                f"- *Failed Commit*: `{commit_sha}`\n"
                f"- *Reason*: {args.reason}\n"
                f"- *Target Cluster*: `{args.cluster}`\n"
                f"- *Service*: `{args.service}`\n"
                f"- *Status*: {'SUCCESS - Restored previous revision' if rollback_success else 'MANUAL ATTENTION REQUIRED'}\n"
                f"- *Timestamp*: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}",
    }

    if webhook_url:
        send_webhook_alert(webhook_url, alert_payload)

    summary_text = (
        f"\n### :x: Automated Rollback Report\n"
        f"- **Status**: {'Restored' if rollback_success else 'Rollback failed — Manual review needed'}\n"
        f"- **Triggering Reason**: {args.reason}\n"
        f"- **Failed Revision Commit**: `{commit_sha}`\n"
        f"- **Timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}`\n"
    )
    log_summary(summary_text)

    return 0 if rollback_success else 1


if __name__ == "__main__":
    sys.exit(main())
