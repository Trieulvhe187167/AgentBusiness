"""Manage benchmarkable RAG production profiles.

The script intentionally does not edit `.env` by default. It exports env
overlays for candidate runs and records promote/rollback decisions in a JSON
state file after golden-eval benchmark evidence exists.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STATE_VERSION = 1
DEFAULT_STATE_PATH = "data/rag_profiles/state.json"


PROFILES: dict[str, dict[str, Any]] = {
    "current_local": {
        "label": "Current local baseline",
        "description": "Keep the existing lightweight local posture: numpy/Chroma fallback and BM25-lite reranking.",
        "requires_reindex": False,
        "env": {
            "RAG_RERANKER_PROVIDER": "bm25_lite",
            "RAG_RETRIEVAL_SOURCE_DIVERSIFICATION_ENABLED": "false",
            "RAG_RETRIEVAL_SOURCE_MAX_CHUNKS_PER_SOURCE": "0",
            "RAG_QDRANT_HYBRID_ENABLED": "false",
        },
        "benchmark_order": 0,
        "changes_index_contract": False,
    },
    "local_cross_encoder": {
        "label": "Local neural reranker candidate",
        "description": "Keep the current vector index and benchmark Qwen3 cross-encoder reranking only.",
        "requires_reindex": False,
        "env": {
            "RAG_RERANKER_PROVIDER": "cross_encoder",
            "RAG_RERANKER_MODEL": "Qwen/Qwen3-Reranker-0.6B",
            "RAG_RERANKER_TOP_N": "50",
            "RAG_RERANKER_BATCH_SIZE": "8",
            "RAG_RERANKER_TIMEOUT_SECONDS": "10",
            "RAG_RERANKER_WEIGHT": "0.85",
            "RAG_RUNTIME_MAX_RERANK_CANDIDATES": "50",
            "RAG_CORRECTIVE_RAG_ENABLED": "false",
        },
        "benchmark_order": 1,
        "changes_index_contract": False,
    },
    "local_corrective_rag": {
        "label": "Local corrective RAG candidate",
        "description": "Keep BM25-lite reranking and benchmark one corrective rewrite attempt for weak retrieval.",
        "requires_reindex": False,
        "env": {
            "RAG_RERANKER_PROVIDER": "bm25_lite",
            "RAG_CORRECTIVE_RAG_ENABLED": "true",
            "RAG_CORRECTIVE_RAG_MAX_ATTEMPTS": "1",
            "RAG_CORRECTIVE_RAG_MIN_SCORE": "0.35",
            "RAG_CORRECTIVE_RAG_MIN_RESULTS": "2",
            "RAG_CORRECTIVE_RAG_REWRITE_TIMEOUT_SECONDS": "8",
            "RAG_CORRECTIVE_RAG_REWRITE_MAX_TOKENS": "128",
        },
        "benchmark_order": 2,
        "changes_index_contract": False,
    },
    "local_cross_encoder_corrective": {
        "label": "Local neural reranker + corrective RAG candidate",
        "description": "Benchmark the combined N+1 candidate without changing embeddings or vector index schema.",
        "requires_reindex": False,
        "env": {
            "RAG_RERANKER_PROVIDER": "cross_encoder",
            "RAG_RERANKER_MODEL": "Qwen/Qwen3-Reranker-0.6B",
            "RAG_RERANKER_TOP_N": "50",
            "RAG_RERANKER_BATCH_SIZE": "8",
            "RAG_RERANKER_TIMEOUT_SECONDS": "10",
            "RAG_RERANKER_WEIGHT": "0.85",
            "RAG_RUNTIME_MAX_RERANK_CANDIDATES": "50",
            "RAG_CORRECTIVE_RAG_ENABLED": "true",
            "RAG_CORRECTIVE_RAG_MAX_ATTEMPTS": "1",
            "RAG_CORRECTIVE_RAG_MIN_SCORE": "0.35",
            "RAG_CORRECTIVE_RAG_MIN_RESULTS": "2",
            "RAG_CORRECTIVE_RAG_REWRITE_TIMEOUT_SECONDS": "8",
            "RAG_CORRECTIVE_RAG_REWRITE_MAX_TOKENS": "128",
        },
        "benchmark_order": 3,
        "changes_index_contract": False,
    },
    "qdrant_dense_bge_m3": {
        "label": "Qdrant dense BGE-M3",
        "description": "Production-like dense retrieval with a multilingual embedding baseline and BM25-lite reranking.",
        "requires_reindex": True,
        "env": {
            "RAG_VECTOR_BACKEND": "qdrant",
            "RAG_QDRANT_COLLECTION_NAME": "kb_chunks_bge_m3_dense_v1",
            "RAG_QDRANT_HYBRID_ENABLED": "false",
            "RAG_EMBEDDING_PROVIDER": "sentence_transformers",
            "RAG_EMBEDDING_MODEL": "BAAI/bge-m3",
            "RAG_EMBEDDING_DIMENSION": "0",
            "RAG_EMBEDDING_TRUST_REMOTE_CODE": "false",
            "RAG_RERANKER_PROVIDER": "bm25_lite",
        },
        "benchmark_order": 10,
        "changes_index_contract": True,
    },
    "qdrant_hybrid_bge_m3": {
        "label": "Qdrant hybrid BGE-M3",
        "description": "Dense+sparse Qdrant retrieval with RRF fusion, source diversification, and BM25-lite reranking.",
        "requires_reindex": True,
        "env": {
            "RAG_VECTOR_BACKEND": "qdrant",
            "RAG_QDRANT_COLLECTION_NAME": "kb_chunks_bge_m3_hybrid_v1",
            "RAG_QDRANT_HYBRID_ENABLED": "true",
            "RAG_QDRANT_SPARSE_MODEL": "Qdrant/bm25",
            "RAG_QDRANT_HYBRID_PREFETCH_K": "50",
            "RAG_QDRANT_HYBRID_MIN_SIMILARITY_THRESHOLD": "0.10",
            "RAG_QDRANT_HYBRID_THRESHOLD_LOW": "0.20",
            "RAG_QDRANT_HYBRID_THRESHOLD_GOOD": "0.35",
            "RAG_EMBEDDING_PROVIDER": "sentence_transformers",
            "RAG_EMBEDDING_MODEL": "BAAI/bge-m3",
            "RAG_EMBEDDING_DIMENSION": "0",
            "RAG_EMBEDDING_TRUST_REMOTE_CODE": "false",
            "RAG_RETRIEVAL_SOURCE_DIVERSIFICATION_ENABLED": "true",
            "RAG_RETRIEVAL_SOURCE_MAX_CHUNKS_PER_SOURCE": "2",
            "RAG_RERANKER_PROVIDER": "bm25_lite",
        },
        "benchmark_order": 11,
        "changes_index_contract": True,
    },
    "qdrant_hybrid_qwen3": {
        "label": "Qdrant hybrid Qwen3 embedding",
        "description": "Hybrid retrieval with Qwen3 embedding instructions and BM25-lite reranking.",
        "requires_reindex": True,
        "env": {
            "RAG_VECTOR_BACKEND": "qdrant",
            "RAG_QDRANT_COLLECTION_NAME": "kb_chunks_qwen3_06b_hybrid_v1",
            "RAG_QDRANT_HYBRID_ENABLED": "true",
            "RAG_QDRANT_SPARSE_MODEL": "Qdrant/bm25",
            "RAG_QDRANT_HYBRID_PREFETCH_K": "50",
            "RAG_QDRANT_HYBRID_MIN_SIMILARITY_THRESHOLD": "0.10",
            "RAG_QDRANT_HYBRID_THRESHOLD_LOW": "0.20",
            "RAG_QDRANT_HYBRID_THRESHOLD_GOOD": "0.35",
            "RAG_EMBEDDING_PROVIDER": "sentence_transformers",
            "RAG_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
            "RAG_EMBEDDING_DIMENSION": "0",
            "RAG_EMBEDDING_TRUST_REMOTE_CODE": "false",
            "RAG_EMBEDDING_QUERY_INSTRUCTION": (
                "Given a business support question in Vietnamese or English, "
                "retrieve relevant internal knowledge base passages."
            ),
            "RAG_RETRIEVAL_SOURCE_DIVERSIFICATION_ENABLED": "true",
            "RAG_RETRIEVAL_SOURCE_MAX_CHUNKS_PER_SOURCE": "2",
            "RAG_RERANKER_PROVIDER": "bm25_lite",
        },
        "benchmark_order": 12,
        "changes_index_contract": True,
    },
    "qdrant_hybrid_qwen3_reranker": {
        "label": "Qdrant hybrid Qwen3 embedding + neural reranker",
        "description": "Hybrid retrieval with Qwen3 embedding and Qwen3 cross-encoder reranking.",
        "requires_reindex": True,
        "env": {
            "RAG_VECTOR_BACKEND": "qdrant",
            "RAG_QDRANT_COLLECTION_NAME": "kb_chunks_qwen3_06b_hybrid_rerank_v1",
            "RAG_QDRANT_HYBRID_ENABLED": "true",
            "RAG_QDRANT_SPARSE_MODEL": "Qdrant/bm25",
            "RAG_QDRANT_HYBRID_PREFETCH_K": "80",
            "RAG_QDRANT_HYBRID_MIN_SIMILARITY_THRESHOLD": "0.10",
            "RAG_QDRANT_HYBRID_THRESHOLD_LOW": "0.20",
            "RAG_QDRANT_HYBRID_THRESHOLD_GOOD": "0.35",
            "RAG_EMBEDDING_PROVIDER": "sentence_transformers",
            "RAG_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
            "RAG_EMBEDDING_DIMENSION": "0",
            "RAG_EMBEDDING_TRUST_REMOTE_CODE": "false",
            "RAG_EMBEDDING_QUERY_INSTRUCTION": (
                "Given a business support question in Vietnamese or English, "
                "retrieve relevant internal knowledge base passages."
            ),
            "RAG_RETRIEVAL_SOURCE_DIVERSIFICATION_ENABLED": "true",
            "RAG_RETRIEVAL_SOURCE_MAX_CHUNKS_PER_SOURCE": "2",
            "RAG_RERANKER_PROVIDER": "cross_encoder",
            "RAG_RERANKER_MODEL": "Qwen/Qwen3-Reranker-0.6B",
            "RAG_RERANKER_TOP_N": "50",
            "RAG_RERANKER_BATCH_SIZE": "8",
            "RAG_RERANKER_TIMEOUT_SECONDS": "10",
            "RAG_RERANKER_WEIGHT": "0.85",
            "RAG_RUNTIME_MAX_RERANK_CANDIDATES": "50",
        },
        "benchmark_order": 13,
        "changes_index_contract": True,
    },
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def get_profile(name: str) -> dict[str, Any]:
    if name not in PROFILES:
        valid = ", ".join(sorted(PROFILES))
        raise SystemExit(f"Unknown profile '{name}'. Valid profiles: {valid}")
    profile = deepcopy(PROFILES[name])
    profile["name"] = name
    return profile


def profile_env_text(name: str) -> str:
    profile = get_profile(name)
    lines = [
        f"# RAG profile: {name}",
        f"# {profile['label']}",
        f"# Requires reindex: {str(profile['requires_reindex']).lower()}",
    ]
    if profile["env"].get("RAG_VECTOR_BACKEND") == "qdrant":
        lines.append("# Set RAG_QDRANT_URL or RAG_QDRANT_PATH for your environment before starting the app.")
    for key in sorted(profile["env"]):
        lines.append(f"{key}={profile['env'][key]}")
    lines.append("")
    return "\n".join(lines)


def load_report(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def gate_status_from_report(report: dict[str, Any] | None) -> str:
    if not report:
        return "unknown"
    summary = report.get("summary") or {}
    return str(summary.get("gate_status") or "unknown")


def run_id_from_report(report: dict[str, Any] | None) -> int | None:
    if not report:
        return None
    raw = (report.get("summary") or {}).get("run_id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def metrics_from_report(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return dict((report.get("summary") or {}).get("metrics") or {})


def coverage_from_report(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    readiness = report.get("readiness") or {}
    coverage = readiness.get("coverage") or {}
    return dict(coverage) if isinstance(coverage, dict) else {}


def coverage_is_representative(report: dict[str, Any] | None) -> bool:
    return bool(coverage_from_report(report).get("representative_ready"))


def default_state() -> dict[str, Any]:
    return {
        "state_version": STATE_VERSION,
        "active_profile": None,
        "previous_profile": None,
        "last_promotion": None,
        "history": [],
    }


def load_state(path: str) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return default_state()
    with state_path.open(encoding="utf-8") as fh:
        state = json.load(fh)
    if not isinstance(state, dict):
        raise SystemExit(f"Invalid RAG profile state file: {path}")
    state.setdefault("state_version", STATE_VERSION)
    state.setdefault("history", [])
    return state


def save_state(path: str, state: dict[str, Any]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def promote_profile(
    *,
    state_path: str,
    profile_name: str,
    candidate_report: str | None,
    baseline_run_id: int | None,
    candidate_run_id: int | None,
    reason: str,
    allow_failed_gate: bool,
    allow_insufficient_golden: bool = False,
) -> dict[str, Any]:
    profile = get_profile(profile_name)
    report = load_report(candidate_report)
    gate_status = gate_status_from_report(report)
    if gate_status == "failed" and not allow_failed_gate:
        raise SystemExit("Candidate gate failed. Re-run with --allow-failed-gate only with an explicit override reason.")
    if profile.get("changes_index_contract") and not coverage_is_representative(report) and not allow_insufficient_golden:
        coverage = coverage_from_report(report)
        raise SystemExit(
            "Representative golden coverage is required before promoting embedding/vector-index profiles. "
            f"reviewed_active_items={coverage.get('reviewed_active_items', 0)}, "
            f"target={coverage.get('target_active_reviewed_items', 80)}, "
            f"missing_buckets={coverage.get('missing_required_buckets', [])}. "
            "Re-run with --allow-insufficient-golden only with an explicit owner override."
        )
    resolved_candidate_run_id = candidate_run_id or run_id_from_report(report)
    if not resolved_candidate_run_id:
        raise SystemExit("candidate run id is required; pass --candidate-run-id or --candidate-report with summary.run_id")
    if not reason.strip():
        raise SystemExit("--reason is required for promotion audit trail")

    state = load_state(state_path)
    previous = state.get("active_profile")
    event = {
        "event": "promote",
        "profile": profile_name,
        "previous_profile": previous,
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": resolved_candidate_run_id,
        "candidate_report": candidate_report,
        "gate_status": gate_status,
        "metrics": metrics_from_report(report),
        "golden_coverage": coverage_from_report(report),
        "requires_reindex": bool(profile["requires_reindex"]),
        "changes_index_contract": bool(profile.get("changes_index_contract")),
        "reason": reason.strip(),
        "created_at": now_iso(),
    }
    state["previous_profile"] = previous
    state["active_profile"] = profile_name
    state["last_promotion"] = event
    state.setdefault("history", []).append(event)
    save_state(state_path, state)
    return event


def rollback_profile(*, state_path: str, reason: str) -> dict[str, Any]:
    if not reason.strip():
        raise SystemExit("--reason is required for rollback audit trail")
    state = load_state(state_path)
    active = state.get("active_profile")
    previous = state.get("previous_profile")
    if not previous:
        raise SystemExit("No previous_profile recorded; cannot rollback")
    event = {
        "event": "rollback",
        "from_profile": active,
        "to_profile": previous,
        "reason": reason.strip(),
        "created_at": now_iso(),
    }
    state["active_profile"] = previous
    state["previous_profile"] = active
    state.setdefault("history", []).append(event)
    save_state(state_path, state)
    return event


def benchmark_plan(base_url: str, kb_id: int, limit: int, baseline_run_id: int | None) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for name, profile in sorted(PROFILES.items(), key=lambda item: int(item[1]["benchmark_order"])):
        report = f"data/rag_profiles/eval_{name}.json"
        args = [
            ".\\.venv\\Scripts\\python.exe",
            "scripts\\run_baseline_eval.py",
            "--base-url",
            base_url,
            "--kb-id",
            str(kb_id),
            "--limit",
            str(limit),
            "--report-output",
            report,
        ]
        if baseline_run_id and name != "current_local":
            args.extend(["--baseline-run-id", str(baseline_run_id), "--fail-on-gate"])
        steps.append(
            {
                "profile": name,
                "label": profile["label"],
                "requires_reindex": profile["requires_reindex"],
                "requires_representative_golden": bool(profile.get("changes_index_contract")),
                "env_overlay": f"data/rag_profiles/{name}.env",
                "report": report,
                "command": " ".join(args),
            }
        )
    return {"base_url": base_url, "kb_id": kb_id, "limit": limit, "steps": steps}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", default=DEFAULT_STATE_PATH, help="RAG profile state JSON path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage RAG profile benchmark/promote/rollback flow")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List built-in profiles")
    list_parser.add_argument("--json", action="store_true", help="Print JSON instead of text")

    show = sub.add_parser("show", help="Show one profile")
    show.add_argument("profile")
    show.add_argument("--format", choices=["json", "env"], default="json")

    write_env = sub.add_parser("write-env", help="Write an env overlay for one profile")
    write_env.add_argument("profile")
    write_env.add_argument("--output", required=True)

    plan = sub.add_parser("plan-benchmark", help="Print benchmark commands for all profiles")
    plan.add_argument("--base-url", default="http://127.0.0.1:8080")
    plan.add_argument("--kb-id", type=int, default=1)
    plan.add_argument("--limit", type=int, default=150)
    plan.add_argument("--baseline-run-id", type=int, default=None)

    promote = sub.add_parser("promote", help="Record a promoted profile after benchmark")
    _add_common(promote)
    promote.add_argument("profile")
    promote.add_argument("--candidate-report", default="")
    promote.add_argument("--baseline-run-id", type=int, default=None)
    promote.add_argument("--candidate-run-id", type=int, default=None)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--allow-failed-gate", action="store_true")
    promote.add_argument("--allow-insufficient-golden", action="store_true")

    rollback = sub.add_parser("rollback", help="Rollback to the previously active profile")
    _add_common(rollback)
    rollback.add_argument("--reason", required=True)

    status = sub.add_parser("status", help="Show active profile state")
    _add_common(status)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list":
        rows = [
            {
                "name": name,
                "label": profile["label"],
                "requires_reindex": profile["requires_reindex"],
                "changes_index_contract": bool(profile.get("changes_index_contract")),
                "benchmark_order": profile["benchmark_order"],
            }
            for name, profile in sorted(PROFILES.items(), key=lambda item: int(item[1]["benchmark_order"]))
        ]
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                marker = "reindex" if row["requires_reindex"] else "no-reindex"
                print(f"{row['name']}: {row['label']} ({marker})")
        return 0
    if args.command == "show":
        if args.format == "env":
            print(profile_env_text(args.profile), end="")
        else:
            print(json.dumps(get_profile(args.profile), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "write-env":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(profile_env_text(args.profile), encoding="utf-8")
        print(json.dumps({"profile": args.profile, "output": str(output.resolve())}, ensure_ascii=False))
        return 0
    if args.command == "plan-benchmark":
        print(json.dumps(benchmark_plan(args.base_url, args.kb_id, args.limit, args.baseline_run_id), ensure_ascii=False, indent=2))
        return 0
    if args.command == "promote":
        event = promote_profile(
            state_path=args.state,
            profile_name=args.profile,
            candidate_report=args.candidate_report or None,
            baseline_run_id=args.baseline_run_id,
            candidate_run_id=args.candidate_run_id,
            reason=args.reason,
            allow_failed_gate=args.allow_failed_gate,
            allow_insufficient_golden=args.allow_insufficient_golden,
        )
        print(json.dumps(event, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "rollback":
        event = rollback_profile(state_path=args.state, reason=args.reason)
        print(json.dumps(event, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        print(json.dumps(load_state(args.state), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
