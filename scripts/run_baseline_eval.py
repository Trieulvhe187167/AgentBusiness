"""Create a reproducible Phase 0 golden-dataset baseline evaluation run.

This script calls the running FastAPI app instead of importing app modules so the
baseline captures the same auth, routing, retrieval, and persistence path used
in production/local deployments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

REVIEWED_TAGS = {"reviewed", "approved", "golden_reviewed", "source_verified"}
SMOKE_TAGS = {"smoke_auto_activated"}
REQUIRED_COVERAGE_BUCKETS = {
    "vi",
    "en",
    "exact",
    "citation",
    "negative",
    "followup",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Phase 0 RAG baseline eval")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL")
    parser.add_argument("--kb-id", type=int, default=1, help="Knowledge base id to benchmark")
    parser.add_argument("--name", default="", help="Optional eval run name")
    parser.add_argument("--limit", type=int, default=150, help="Max golden items to evaluate")
    parser.add_argument("--min-golden-items", type=int, default=80, help="Recommended minimum golden item count")
    parser.add_argument("--allow-small", action="store_true", help="Run even if golden dataset is below the minimum")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect golden dataset readiness; do not create an eval run")
    parser.add_argument("--llm-judge", action="store_true", help="Enable optional LLM-as-judge for this run")
    parser.add_argument("--baseline-run-id", type=int, default=None, help="Compare this run against an existing baseline")
    parser.add_argument("--max-metric-drop", type=float, default=0.05, help="Allowed metric drop when comparing to a baseline")
    parser.add_argument("--report-output", default="", help="Optional JSON report path")
    parser.add_argument("--fail-on-gate", action="store_true", help="Exit non-zero when comparison gate fails")
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


def _norm_items(items: list[Any]) -> set[str]:
    out: set[str] = set()
    for item in items or []:
        text = str(item or "").strip().lower()
        if text:
            out.add(text)
    return out


def _coverage_bucket_names(item: dict[str, Any]) -> set[str]:
    values = _norm_items(item.get("tags") or []) | _norm_items(item.get("expected_categories") or [])
    buckets: set[str] = set()
    joined = " ".join(values)
    if "policy_vi" in values or " vi" in f" {joined}" or "vietnamese" in joined:
        buckets.add("vi")
    if "policy_en" in values or " en" in f" {joined}" or "english" in joined:
        buckets.add("en")
    if "exact_id" in values or "exact" in joined or "code" in joined:
        buckets.add("exact")
    if "citation" in joined or item.get("expected_source_file_ids") or item.get("expected_source_file_id"):
        buckets.add("citation")
    if "negative" in joined or "no_answer" in joined or "unknown" in joined:
        buckets.add("negative")
    if "followup" in joined or "follow_up" in joined:
        buckets.add("followup")
    if "mixed" in joined or "vi_en" in joined:
        buckets.add("mixed")
    return buckets


def _golden_coverage(items: list[dict[str, Any]], *, target: int) -> dict[str, Any]:
    active_items = [item for item in items if bool(item.get("active", True))]
    reviewed_items = []
    smoke_items = []
    bucket_counts = {name: 0 for name in sorted(REQUIRED_COVERAGE_BUCKETS | {"mixed"})}
    for item in active_items:
        tags = _norm_items(item.get("tags") or [])
        if tags & SMOKE_TAGS:
            smoke_items.append(item)
        if tags & REVIEWED_TAGS:
            reviewed_items.append(item)
        for bucket in _coverage_bucket_names(item):
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    missing = sorted(bucket for bucket in REQUIRED_COVERAGE_BUCKETS if bucket_counts.get(bucket, 0) <= 0)
    return {
        "target_active_reviewed_items": int(target),
        "active_items": len(active_items),
        "reviewed_active_items": len(reviewed_items),
        "smoke_active_items": len(smoke_items),
        "bucket_counts": bucket_counts,
        "missing_required_buckets": missing,
        "representative_ready": len(reviewed_items) >= int(target) and not missing,
    }


def _print_json(label: str, payload: Any) -> None:
    print(f"\n{label}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _write_report(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


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
        dataset_items = dataset_payload.get("items") or []
        total = int(dataset_payload.get("total") or 0)
        coverage = _golden_coverage(dataset_items if isinstance(dataset_items, list) else [], target=args.min_golden_items)
        readiness = {
            "kb_id": args.kb_id,
            "active_golden_items": total,
            "recommended_minimum": args.min_golden_items,
            "ready": total >= args.min_golden_items,
            "will_run": not args.dry_run and (total >= args.min_golden_items or args.allow_small),
            "coverage": coverage,
        }
        _print_json("Golden dataset readiness", readiness)
        if args.dry_run:
            return 0
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
            "baseline_run_id": args.baseline_run_id,
            "max_metric_drop": args.max_metric_drop,
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
        "comparison": result.get("comparison") or {},
    }
    report = {
        "readiness": readiness,
        "summary": summary,
        "rag_config_snapshot": (result.get("config") or {}).get("rag_config_snapshot") or {},
        "failed_or_warn_examples": [
            {
                "golden_item_id": item.get("golden_item_id"),
                "verdict": item.get("verdict"),
                "score": item.get("score"),
                "reason": item.get("reason"),
                "question": item.get("user_message"),
                "trace": item.get("trace") or {},
                "retrieved": item.get("retrieved") or [],
                "citations": item.get("citations") or [],
            }
            for item in (result.get("results") or [])
            if item.get("verdict") in {"fail", "warn"}
        ][:25],
        "diagnostics": {
            "top_latency_examples": sorted(
                [
                    {
                        "golden_item_id": item.get("golden_item_id"),
                        "latency_ms": item.get("latency_ms"),
                        "verdict": item.get("verdict"),
                        "question": item.get("user_message"),
                        "trace": item.get("trace") or {},
                    }
                    for item in (result.get("results") or [])
                    if item.get("latency_ms") is not None
                ],
                key=lambda item: float(item.get("latency_ms") or 0),
                reverse=True,
            )[:10],
            "retrieval_miss_examples": [
                {
                    "golden_item_id": item.get("golden_item_id"),
                    "verdict": item.get("verdict"),
                    "question": item.get("user_message"),
                    "trace": item.get("trace") or {},
                }
                for item in (result.get("results") or [])
                if (item.get("trace") or {}).get("explain", {}).get("retrieval_failed")
            ][:10],
            "citation_mismatch_examples": [
                {
                    "golden_item_id": item.get("golden_item_id"),
                    "verdict": item.get("verdict"),
                    "question": item.get("user_message"),
                    "trace": item.get("trace") or {},
                }
                for item in (result.get("results") or [])
                if (item.get("trace") or {}).get("explain", {}).get("citation_mismatch")
            ][:10],
        },
    }
    _print_json("Baseline summary", summary)
    _print_json("Captured RAG config snapshot", (result.get("config") or {}).get("rag_config_snapshot") or {})
    _write_report(args.report_output, report)
    if args.report_output:
        print(f"\nReport written: {args.report_output}")
    if args.fail_on_gate and summary.get("gate_status") == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
