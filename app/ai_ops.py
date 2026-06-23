"""
AI operations summaries and safe replay helpers.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.database import fetch_all, fetch_one, fetch_one_sync

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)")
_SENSITIVE_KEYS = {
    "contact",
    "email",
    "phone",
    "user_id",
    "created_by_user_id",
    "approved_by_user_id",
    "executed_by_user_id",
    "tenant_id",
    "org_id",
}


def _period_modifier(days: int) -> str:
    return f"-{int(days)} days"


def _kb_clause(alias: str, kb_id: int | None) -> tuple[str, tuple[int, ...]]:
    if kb_id is None:
        return "", ()
    return f" AND {alias}.kb_id = ?", (int(kb_id),)


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _percentile(values: list[int], percentile: float) -> int | None:
    clean = sorted(int(value) for value in values if value is not None)
    if not clean:
        return None
    index = min(len(clean) - 1, max(0, round((len(clean) - 1) * percentile)))
    return clean[index]


def _safe_json_loads(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def _redact_text(value: str) -> str:
    text = _EMAIL_RE.sub("[redacted-email]", value)
    return _PHONE_RE.sub("[redacted-phone]", text)


def redact_payload(value: Any, *, max_text: int = 500) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _SENSITIVE_KEYS or key_text.lower().endswith("_user_id"):
                redacted[key_text] = "[redacted]"
            else:
                redacted[key_text] = redact_payload(item, max_text=max_text)
        return redacted
    if isinstance(value, list):
        return [redact_payload(item, max_text=max_text) for item in value[:50]]
    if isinstance(value, str):
        return _redact_text(value)[:max_text]
    return value


def _status_from_alerts(alerts: list[dict[str, Any]]) -> str:
    severities = {str(item.get("severity") or "info") for item in alerts}
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    return "healthy"


async def build_ai_ops_summary(days: int = 7, kb_id: int | None = None) -> dict[str, Any]:
    period = _period_modifier(days)
    chat_kb_sql, chat_kb_params = _kb_clause("cl", kb_id)
    tool_kb_sql, tool_kb_params = _kb_clause("tal", kb_id)
    pending_kb_sql, pending_kb_params = _kb_clause("pa", kb_id)
    eval_kb_sql, eval_kb_params = _kb_clause("aer", kb_id)

    chat_row = await fetch_one(
        f"""
        SELECT
            COUNT(*) AS chat_count,
            SUM(COALESCE(llm_input_tokens, 0)) AS input_tokens,
            SUM(COALESCE(llm_output_tokens, 0)) AS output_tokens,
            SUM(COALESCE(llm_total_tokens, 0)) AS total_tokens,
            SUM(COALESCE(llm_cached_tokens, 0)) AS cached_tokens,
            SUM(CASE WHEN mode = 'fallback' THEN 1 ELSE 0 END) AS fallback_count
        FROM chat_logs cl
        WHERE datetime(cl.created_at) >= datetime('now', ?)
        {chat_kb_sql}
        """,
        (period, *chat_kb_params),
    )
    latency_rows = await fetch_all(
        f"""
        SELECT latency_ms
        FROM chat_logs cl
        WHERE datetime(cl.created_at) >= datetime('now', ?)
          AND latency_ms IS NOT NULL
        {chat_kb_sql}
        """,
        (period, *chat_kb_params),
    )
    tool_rows = await fetch_all(
        f"""
        SELECT
            COALESCE(tool_name, 'unknown') AS tool_name,
            COUNT(*) AS calls,
            SUM(CASE WHEN tool_status NOT IN ('success', 'clarify') THEN 1 ELSE 0 END) AS failures,
            AVG(latency_ms) AS avg_latency_ms
        FROM tool_audit_logs tal
        WHERE datetime(tal.created_at) >= datetime('now', ?)
        {tool_kb_sql}
        GROUP BY COALESCE(tool_name, 'unknown')
        ORDER BY failures DESC, calls DESC
        LIMIT 8
        """,
        (period, *tool_kb_params),
    )
    pending_row = await fetch_one(
        f"""
        SELECT
            COUNT(*) AS open_count,
            AVG((julianday('now') - julianday(pa.created_at)) * 24.0) AS avg_open_hours,
            MAX((julianday('now') - julianday(pa.created_at)) * 24.0) AS max_open_hours
        FROM pending_actions pa
        WHERE pa.status IN ('draft', 'approved')
        {pending_kb_sql}
        """,
        pending_kb_params,
    )
    eval_rows = await fetch_all(
        f"""
        SELECT
            aer.id, aer.name, aer.source, aer.kb_id, aer.sample_size,
            aer.pass_count, aer.warn_count, aer.fail_count, aer.avg_score,
            aer.gate_status, aer.metrics_json, aer.comparison_json,
            aer.created_at, aer.completed_at
        FROM agent_eval_runs aer
        WHERE datetime(aer.created_at) >= datetime('now', ?)
        {eval_kb_sql}
        ORDER BY aer.created_at DESC, aer.id DESC
        LIMIT 5
        """,
        (period, *eval_kb_params),
    )

    input_tokens = int((chat_row or {}).get("input_tokens") or 0)
    output_tokens = int((chat_row or {}).get("output_tokens") or 0)
    total_tokens = int((chat_row or {}).get("total_tokens") or 0)
    cached_tokens = int((chat_row or {}).get("cached_tokens") or 0)
    latency_values = [int(row.get("latency_ms") or 0) for row in latency_rows]
    tool_calls = sum(int(row.get("calls") or 0) for row in tool_rows)
    tool_failures = sum(int(row.get("failures") or 0) for row in tool_rows)
    cache_rate = _rate(cached_tokens, input_tokens)
    tool_error_rate = _rate(tool_failures, tool_calls)
    p95_latency_ms = _percentile(latency_values, 0.95)
    latest_eval = dict(eval_rows[0]) if eval_rows else None

    alerts: list[dict[str, Any]] = []
    if p95_latency_ms and p95_latency_ms > 3000:
        alerts.append({"severity": "warning", "code": "latency_p95_high", "message": "Chat p95 latency is above 3 seconds."})
    if tool_error_rate is not None and tool_error_rate > 0.05:
        alerts.append({"severity": "warning", "code": "tool_error_budget", "message": "Tool error rate exceeded the 5% budget."})
    if latest_eval and latest_eval.get("gate_status") == "failed":
        alerts.append({"severity": "critical", "code": "eval_gate_failed", "message": "Latest evaluation gate failed."})
    if int((pending_row or {}).get("open_count") or 0) > 0:
        alerts.append({"severity": "warning", "code": "approval_backlog", "message": "Pending approvals are waiting for review."})
    if input_tokens >= 1000 and (cache_rate is None or cache_rate < 0.25):
        alerts.append({"severity": "warning", "code": "cache_reuse_low", "message": "Input token cache reuse is below 25%."})

    eval_trend = [
        {
            "id": int(row["id"]),
            "name": row.get("name") or f"Run {row['id']}",
            "source": row.get("source") or "chat_logs",
            "kb_id": row.get("kb_id"),
            "sample_size": int(row.get("sample_size") or 0),
            "avg_score": round(float(row["avg_score"]), 2) if row.get("avg_score") is not None else None,
            "gate_status": row.get("gate_status") or "not_compared",
            "pass_count": int(row.get("pass_count") or 0),
            "warn_count": int(row.get("warn_count") or 0),
            "fail_count": int(row.get("fail_count") or 0),
            "metrics": _safe_json_loads(row.get("metrics_json"), {}),
            "created_at": row.get("created_at"),
            "completed_at": row.get("completed_at"),
        }
        for row in eval_rows
    ]

    return {
        "period_days": int(days),
        "kb_id": kb_id,
        "status": _status_from_alerts(alerts),
        "alerts": alerts,
        "cost": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": cached_tokens,
            "cache_reuse_rate": cache_rate,
            "billable_input_token_estimate": max(input_tokens - cached_tokens, 0),
        },
        "latency": {
            "sample_size": len(latency_values),
            "p50_ms": _percentile(latency_values, 0.50),
            "p95_ms": p95_latency_ms,
            "max_ms": max(latency_values) if latency_values else None,
        },
        "tooling": {
            "calls": tool_calls,
            "failures": tool_failures,
            "error_rate": tool_error_rate,
            "top_tools": [
                {
                    "tool_name": row.get("tool_name") or "unknown",
                    "calls": int(row.get("calls") or 0),
                    "failures": int(row.get("failures") or 0),
                    "avg_latency_ms": round(float(row["avg_latency_ms"]), 2) if row.get("avg_latency_ms") is not None else None,
                }
                for row in tool_rows
            ],
        },
        "approvals": {
            "open_count": int((pending_row or {}).get("open_count") or 0),
            "avg_open_hours": round(float((pending_row or {}).get("avg_open_hours") or 0), 2),
            "max_open_hours": round(float((pending_row or {}).get("max_open_hours") or 0), 2),
        },
        "eval_trend": eval_trend,
        "redaction_policy": sorted(_SENSITIVE_KEYS),
    }


def replay_chat_log(chat_log_id: int, *, mode: str = "retrieval_only", top_k: int = 5) -> dict[str, Any]:
    if mode != "retrieval_only":
        raise ValueError("Only retrieval_only replay is supported")

    row = fetch_one_sync(
        """
        SELECT *
        FROM chat_logs
        WHERE id = ?
        """,
        (int(chat_log_id),),
    )
    if not row:
        raise LookupError(f"Chat log not found: {chat_log_id}")

    query = row.get("merged_query") or row.get("user_message") or ""
    auth_context = {
        "user_id": row.get("user_id"),
        "roles": _safe_json_loads(row.get("roles_json"), []),
        "channel": row.get("channel"),
        "tenant_id": row.get("tenant_id"),
        "org_id": row.get("org_id"),
    }

    from app.rag import decide_mode, retrieve

    started = time.perf_counter()
    results = retrieve(
        query,
        top_k=max(1, min(int(top_k), 20)),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        auth_context=auth_context,
        runtime_context={"disable_corrective_rag": True},
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    top_score = float(results[0].get("similarity", 0.0)) if results else 0.0

    return {
        "mode": mode,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": elapsed_ms,
        "chat_log": redact_payload(
            {
                "id": row["id"],
                "request_id": row.get("request_id"),
                "session_id": row.get("session_id"),
                "user_id": row.get("user_id"),
                "channel": row.get("channel"),
                "tenant_id": row.get("tenant_id"),
                "org_id": row.get("org_id"),
                "kb_id": row.get("kb_id"),
                "kb_key": row.get("kb_key"),
                "original_mode": row.get("mode"),
                "original_top_score": row.get("top_score"),
                "original_latency_ms": row.get("latency_ms"),
                "llm_provider": row.get("llm_provider"),
                "llm_input_tokens": int(row.get("llm_input_tokens") or 0),
                "llm_output_tokens": int(row.get("llm_output_tokens") or 0),
                "llm_cached_tokens": int(row.get("llm_cached_tokens") or 0),
                "user_message": row.get("user_message") or "",
                "answer_text": row.get("answer_text") or "",
                "created_at": row.get("created_at"),
            },
            max_text=600,
        ),
        "query": redact_payload(query, max_text=400),
        "top_score": round(top_score, 4),
        "predicted_mode": decide_mode(top_score),
        "thresholds": {
            "good": settings.threshold_good,
            "low": settings.threshold_low,
            "min": settings.min_similarity_threshold,
        },
        "results": [
            redact_payload(
                {
                    "rank": idx + 1,
                    "similarity": round(float(item.get("similarity", 0.0)), 4),
                    "retrieval_score": round(float(item.get("retrieval_score", item.get("similarity", 0.0))), 4),
                    "final_score": round(float(item.get("final_score", item.get("similarity", 0.0))), 4),
                    "reranker_provider": item.get("reranker_provider"),
                    "filename": item.get("filename") or "",
                    "row_num": item.get("row_num"),
                    "category": item.get("category") or "",
                    "lang": item.get("lang") or "",
                    "snippet": item.get("text") or item.get("content_preview") or "",
                },
                max_text=300,
            )
            for idx, item in enumerate(results)
        ],
    }
