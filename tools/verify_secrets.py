#!/usr/bin/env python3
"""Secret & Rollback Validation Tool for GitHub Actions & AWS ECS Pipeline.

Audits required environment variables and GitHub Secrets for:
1. Container Build & Push (AWS ECR authentication).
2. Deployment Orchestration (AWS ECS service update & task definition registration).
3. Post-Deployment Verification (Target URL, API authentication).
4. Automated Rollback & Incident Alerting (Rollback webhooks & revision management).
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SECRET_DEFINITIONS = [
    {
        "secret": "AWS_ACCESS_KEY_ID",
        "category": "Deployment / Cloud",
        "required": True,
        "description": "IAM user/role access key with ECR and ECS deployment privileges.",
    },
    {
        "secret": "AWS_SECRET_ACCESS_KEY",
        "category": "Deployment / Cloud",
        "required": True,
        "description": "IAM user/role secret access key.",
    },
    {
        "secret": "AWS_REGION",
        "category": "Deployment / Cloud",
        "required": False,
        "default": "us-east-1",
        "description": "Target AWS region for ECR registry and ECS cluster.",
    },
    {
        "secret": "ECR_REPOSITORY",
        "category": "Build / Registry",
        "required": True,
        "description": "Amazon ECR repository name (e.g. 'multimodal-rag-engine').",
    },
    {
        "secret": "ECS_CLUSTER",
        "category": "Deployment / Runtime",
        "required": True,
        "description": "AWS ECS Cluster name hosting the production tasks.",
    },
    {
        "secret": "ECS_SERVICE",
        "category": "Deployment / Runtime",
        "required": True,
        "description": "AWS ECS Service name managing the task instances.",
    },
    {
        "secret": "ECS_TASK_DEFINITION",
        "category": "Deployment / Runtime",
        "required": False,
        "default": "multimodal-rag-task",
        "description": "ECS Task Definition family name for container revision registration.",
    },
    {
        "secret": "PRODUCTION_URL",
        "category": "Health & Verification",
        "required": True,
        "description": "Public HTTPS / internal DNS endpoint of the live deployed service for health verification.",
    },
    {
        "secret": "API_KEY",
        "category": "Security & Auth",
        "required": False,
        "description": "Application X-API-Key passed to verify protected endpoints (/api/v1/query).",
    },
    {
        "secret": "ROLLBACK_WEBHOOK",
        "category": "Rollback & Alerting",
        "required": False,
        "description": "Slack/Teams/OpsGenie webhook URL to broadcast automated rollback notifications.",
    },
    {
        "secret": "GEMINI_API_KEY",
        "category": "VLM / LLM Provider",
        "required": False,
        "description": "Google Gemini Vision API key (optional if using OpenAI or synthetic fallback).",
    },
    {
        "secret": "OPENAI_API_KEY",
        "category": "VLM / LLM Provider",
        "required": False,
        "description": "OpenAI Vision API key (optional if using Gemini or synthetic fallback).",
    },
]


def audit_secrets(check_env: bool = False) -> tuple[int, list[dict]]:
    """Audit secret definitions and report their availability if check_env is True."""
    results = []
    missing_required = 0

    for item in SECRET_DEFINITIONS:
        name = item["secret"]
        val = os.getenv(name)
        present = bool(val and val.strip())

        status = "PRESENT" if present else ("MISSING" if item["required"] else "OPTIONAL")
        if item["required"] and not present:
            missing_required += 1

        results.append({
            "secret": name,
            "category": item["category"],
            "required": item["required"],
            "status": status,
            "description": item["description"],
            "default": item.get("default", "N/A"),
        })

    return missing_required, results


def format_table(results: list[dict]) -> str:
    """Format audit results as a terminal-friendly table."""
    headers = f"{'SECRET NAME':<25} | {'CATEGORY':<22} | {'REQUIRED':<9} | {'STATUS':<9} | {'DEFAULT / NOTE':<15}"
    separator = "-" * len(headers)
    lines = [separator, headers, separator]
    for r in results:
        req_str = "YES" if r["required"] else "NO"
        lines.append(
            f"{r['secret']:<25} | {r['category']:<22} | {req_str:<9} | {r['status']:<9} | {r['default']:<15}"
        )
    lines.append(separator)
    return "\n".join(lines)


def main() -> int:
    """Print the secret mapping and check whether they are configured in the current environment."""
    print("=" * 75)
    print("  GITHUB REPOSITORY SECRETS & ROLLBACK CONFIGURATION MATRIX")
    print("=" * 75)

    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    missing_req, results = audit_secrets(check_env=True)

    print(format_table(results))

    print("\n[+] Secret Integration Guidance for GitHub Actions:")
    print("    1. Go to: GitHub Repository -> Settings -> Secrets and variables -> Actions")
    print("    2. Add the required AWS credentials and target cluster secrets.")
    print("    3. If running locally, you can export these or test with simulated deployment.")

    if is_ci and missing_req > 0:
        print(f"\n[-] CI Environment Error: {missing_req} required secret(s) are missing!")
        return 1

    print("\n[+] Verification Matrix generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
