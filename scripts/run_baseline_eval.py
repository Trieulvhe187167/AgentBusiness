"""Create a reproducible Phase 0 golden-dataset baseline evaluation run.

This script calls the running FastAPI app instead of importing app modules so the
baseline captures the same auth, routing, retrieval, and persistence path used
in production/local deployments.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import httpx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Phase 0 RAG baseline eval")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL")
    parser.add_argument("--kb-id", type=int, default=1, help="Knowledge base id to benchmark")
    parser.add_argument("--name", default="", help="Optional eval run name")
    parser.add_argument("--limit", type=int, default=150, help="Max golden items to evaluate")
    parser.add_argument("--min-golden-items", type=int, default=80, help="Recommended minimum golden item count")
    parser.add_argument("--allow-small", action="store_true", help="Run even if golden dataset is below the minimum")
    parser.add_argument("--llm-judge", action="store_true", help="Enable optional LLM-as-judge for this run")
    parser.add_argument("--user-id", default="baseline-runner", help="Admin user id header")
    parser.add_argument("--tenant-id", default="", help="Optional tenant id header")
    parser.add_argument("--org-id", default="", help="Optional org id header")
    return parser.parse_args()


def _headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {
        "X-User-Id": args.user_id,
        "X-Roles": "admin",
        "X-Channel": "admin",
    }
    if args.tenant_id:
        headers["X-Tenant-Id"] = args.tenant_id
    if args.org_id:
        headers["X-Org-Id"] = args.org_id
    return headers


def _print_json(label: str, payload: Any) -> None:
    print(f"\n{label}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    args = _parse_args()
    base_url = args.base_url.rstrip("/")
    headers = _headers(args)

    with httpx.Client(base_url=base_url, headers=headers, timeout=120) as client:
        dataset = client.get(
            "/api/admin/evaluations/golden-dataset",
            params={"kb_id": args.kb_id, "active": "true", "limit": 500},
        )
        dataset.raise_for_status()
        dataset_payload = dataset.json()
        total = int(dataset_payload.get("total") or 0)
        if total < args.min_golden_items and not args.allow_small:
            raise SystemExit(
                "Golden dataset is too small for a reliable baseline: "
                f"{total}/{args.min_golden_items}. Add more items or pass --allow-small for a smoke baseline."
            )

        name = args.name or f"Phase 0 baseline - KB {args.kb_id}"
        run_payload = {
            "name": name,
            "source": "golden_dataset",
            "kb_id": args.kb_id,
            "limit": args.limit,
            "llm_judge": args.llm_judge,
        }
        run = client.post("/api/admin/evaluations/runs", json=run_payload)
        run.raise_for_status()
        result = run.json()

    summary = {
        "run_id": result.get("id"),
        "name": result.get("name"),
        "source": result.get("source"),
        "kb_id": result.get("kb_id"),
        "sample_size": result.get("sample_size"),
        "pass_count": result.get("pass_count"),
        "warn_count": result.get("warn_count"),
        "fail_count": result.get("fail_count"),
        "avg_score": result.get("avg_score"),
        "gate_status": result.get("gate_status"),
        "baseline_run_id": result.get("baseline_run_id"),
        "metrics": result.get("metrics") or {},
    }
    _print_json("Baseline summary", summary)
    _print_json("Captured RAG config snapshot", (result.get("config") or {}).get("rag_config_snapshot") or {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
