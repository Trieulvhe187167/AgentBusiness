"""
Knowledge gap capture, aggregation, and proactive admin alerts.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.embeddings import embed_texts
from app.models import AuthContext, RequestContext

_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_ACTIVE_STATUSES = {"new", "triaged", "source_needed", "patch_pending"}
_STATUS_ALIASES = {"open": "new", "suggested": "patch_pending", "resolved": "fixed"}
_PRIORITIES = {"P0", "P1", "P2", "P3"}


def _strip_accents(value: str) -> str:
    text = value.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_gap_query(query: str) -> str:
    text = _strip_accents(str(query or "").casefold())
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _query_hash(normalized_query: str) -> str:
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def _cluster_key(normalized_query: str) -> str:
    return _query_hash(normalized_query)[:20]


def _normalize_status(status: str | None, *, default: str = "new") -> str:
    normalized = str(status or default).strip().lower()
    return _STATUS_ALIASES.get(normalized, normalized)


def _normalize_priority(priority: str | None, *, default: str = "P2") -> str:
    normalized = str(priority or default).strip().upper()
    return normalized if normalized in _PRIORITIES else default


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _due_date_for_priority(priority: str) -> str:
    days = {"P0": 1, "P1": 3, "P2": 7, "P3": 14}.get(priority, 7)
    return (datetime.now(timezone.utc) + timedelta(days=days)).date().isoformat()


def _priority_for_gap(*, mode: str, top_score: float | None, threshold: float) -> str:
    normalized = str(mode or "").lower()
    if normalized == "fallback":
        return "P1"
    if normalized.startswith("feedback_down"):
        return "P1"
    if top_score is not None and float(top_score) < float(threshold) * 0.5:
        return "P1"
    return "P2"


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for idx, value in enumerate(left):
        other = float(right[idx])
        current = float(value)
        dot += current * other
        left_norm += current * current
        right_norm += other * other
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def _resolve_semantic_cluster_key(
    *,
    query: str,
    normalized_query: str,
    kb_id: int | None,
    status: str = "new",
) -> str:
    default_key = _cluster_key(normalized_query)
    if not settings.knowledge_gap_semantic_clustering_enabled:
        return default_key

    rows = fetch_all_sync(
        """
        SELECT
            cluster_key,
            normalized_query,
            query,
            COUNT(*) AS count,
            MAX(created_at) AS last_seen_at
        FROM knowledge_gaps
        WHERE status = ?
          AND ((kb_id IS NULL AND ? IS NULL) OR kb_id = ?)
        GROUP BY cluster_key
        ORDER BY count DESC, last_seen_at DESC
        LIMIT 100
        """,
        (status, kb_id, kb_id),
    )
    if not rows:
        return default_key
    for row in rows:
        if row.get("normalized_query") == normalized_query:
            return str(row["cluster_key"])

    candidates = [str(row.get("normalized_query") or row.get("query") or "") for row in rows]
    candidates = [item for item in candidates if item]
    if not candidates:
        return default_key
    try:
        vectors = embed_texts([normalized_query, *candidates], is_query=True)
    except Exception:
        return default_key
    if len(vectors) != len(candidates) + 1:
        return default_key

    query_vector = vectors[0]
    best_idx = -1
    best_score = 0.0
    for idx, vector in enumerate(vectors[1:]):
        score = _cosine_similarity(query_vector, vector)
        if score > best_score:
            best_idx = idx
            best_score = score

    if best_idx >= 0 and best_score >= float(settings.knowledge_gap_semantic_similarity_threshold):
        return str(rows[best_idx]["cluster_key"])
    return default_key


def should_record_gap(*, mode: str, top_score: float | None, threshold: float | None = None) -> bool:
    score_threshold = settings.knowledge_gap_score_threshold if threshold is None else threshold
    normalized_mode = str(mode or "").lower()
    if normalized_mode == "fallback" or normalized_mode.startswith("feedback_down"):
        return True
    if top_score is None:
        return False
    return float(top_score) < float(score_threshold)


def record_knowledge_gap(
    *,
    chat_log_id: int | None,
    query: str,
    mode: str,
    top_score: float | None,
    session_id: str | None,
    threshold: float | None = None,
    context: dict[str, Any] | None = None,
) -> int | None:
    normalized_query = normalize_gap_query(query)
    if not normalized_query:
        return None
    score_threshold = settings.knowledge_gap_score_threshold if threshold is None else threshold
    if not should_record_gap(mode=mode, top_score=top_score, threshold=score_threshold):
        return None

    ctx = context or {}
    auth = ctx.get("auth") or {}
    now = utcnow_iso()
    priority = _normalize_priority(ctx.get("priority"), default=_priority_for_gap(mode=mode, top_score=top_score, threshold=float(score_threshold)))
    due_date = ctx.get("due_date") or _due_date_for_priority(priority)
    query_hash = _query_hash(normalized_query)
    cluster_key = _resolve_semantic_cluster_key(
        query=query,
        normalized_query=normalized_query,
        kb_id=ctx.get("kb_id"),
    )
    existing_cluster = fetch_one_sync(
        """
        SELECT status, owner_user_id, priority, due_date, status_reason
        FROM knowledge_gaps
        WHERE cluster_key = ?
          AND status IN ('new', 'triaged', 'source_needed', 'patch_pending')
          AND ((kb_id IS NULL AND ? IS NULL) OR kb_id = ?)
        ORDER BY updated_at DESC, id DESC
        LIMIT 1
        """,
        (cluster_key, ctx.get("kb_id"), ctx.get("kb_id")),
    )
    status = str((existing_cluster or {}).get("status") or "new")
    owner_user_id = ctx.get("owner_user_id") if ctx.get("owner_user_id") is not None else (existing_cluster or {}).get("owner_user_id")
    priority = _normalize_priority(ctx.get("priority") or (existing_cluster or {}).get("priority"), default=priority)
    due_date = ctx.get("due_date") or (existing_cluster or {}).get("due_date") or due_date
    status_reason = ctx.get("status_reason") if ctx.get("status_reason") is not None else (existing_cluster or {}).get("status_reason")
    gap_id = execute_sync(
        """
        INSERT INTO knowledge_gaps (
            chat_log_id, query, normalized_query, query_hash, mode, top_score,
            threshold, kb_id, kb_key, session_id, tenant_id, org_id,
            cluster_key, status, owner_user_id, priority, due_date, status_reason,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chat_log_id,
            query,
            normalized_query,
            query_hash,
            str(mode or "unknown"),
            float(top_score) if top_score is not None else None,
            float(score_threshold),
            ctx.get("kb_id"),
            ctx.get("kb_key"),
            session_id,
            auth.get("tenant_id"),
            auth.get("org_id"),
            cluster_key,
            status,
            owner_user_id,
            priority,
            due_date,
            status_reason,
            now,
            now,
        ),
    )
    _maybe_create_repeated_gap_alert(
        cluster_key=cluster_key,
        representative_query=query,
        kb_id=ctx.get("kb_id"),
        kb_key=ctx.get("kb_key"),
        tenant_id=auth.get("tenant_id"),
        org_id=auth.get("org_id"),
    )
    return int(gap_id or 0)


def _maybe_create_repeated_gap_alert(
    *,
    cluster_key: str,
    representative_query: str,
    kb_id: int | None,
    kb_key: str | None,
    tenant_id: str | None,
    org_id: str | None,
) -> None:
    repeat_count = max(1, int(settings.knowledge_gap_alert_repeat_count))
    row = fetch_one_sync(
        """
        SELECT COUNT(*) AS count
        FROM knowledge_gaps
        WHERE cluster_key = ?
          AND status IN ('new', 'triaged', 'source_needed', 'patch_pending')
          AND ((kb_id IS NULL AND ? IS NULL) OR kb_id = ?)
        """,
        (cluster_key, kb_id, kb_id),
    )
    count = int((row or {}).get("count") or 0)
    if count < repeat_count:
        return

    entity_id = f"{kb_id or 'global'}:{cluster_key}"
    existing = fetch_one_sync(
        """
        SELECT id FROM notifications
        WHERE event_type = 'knowledge_gap.repeated'
          AND entity_type = 'knowledge_gap_cluster'
          AND entity_id = ?
        LIMIT 1
        """,
        (entity_id,),
    )
    if existing:
        return

    try:
        from app.notifications import create_notification

        create_notification(
            event_type="knowledge_gap.repeated",
            severity="warning",
            title="Repeated knowledge gap detected",
            message=f"{count} unanswered or low-confidence question(s) matched this gap.",
            entity_type="knowledge_gap_cluster",
            entity_id=entity_id,
            payload={
                "cluster_key": cluster_key,
                "count": count,
                "representative_query": representative_query,
                "suggested_action": "create_faq_entry",
            },
            context=RequestContext(
                request_id=f"knowledge-gap-{cluster_key}",
                kb_id=kb_id,
                kb_key=kb_key,
                auth=AuthContext(
                    user_id="knowledge-gap-monitor",
                    roles=["admin"],
                    channel="system",
                    tenant_id=tenant_id,
                    org_id=org_id,
                ),
            ),
        )
    except Exception:
        return


def _sample_queries(raw: str | None) -> list[str]:
    if not raw:
        return []
    samples: list[str] = []
    seen: set[str] = set()
    for item in raw.split("\n"):
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        samples.append(cleaned)
        if len(samples) >= 5:
            break
    return samples


def list_knowledge_gap_clusters(
    *,
    days: int = 7,
    kb_id: int | None = None,
    status: str = "new",
    limit: int = 20,
) -> dict[str, Any]:
    clauses = ["datetime(created_at) >= datetime('now', ?)"]
    params: list[Any] = [f"-{max(1, int(days))} days"]
    if kb_id is not None:
        clauses.append("kb_id = ?")
        params.append(int(kb_id))
    normalized_status = _normalize_status(status, default="new") if status else ""
    if normalized_status:
        clauses.append("status = ?")
        params.append(normalized_status)

    where_sql = " AND ".join(clauses)
    rows = fetch_all_sync(
        f"""
        WITH scoped AS (
            SELECT *
            FROM knowledge_gaps
            WHERE {where_sql}
        ),
        grouped AS (
            SELECT
                cluster_key,
                kb_id,
                kb_key,
                status,
                owner_user_id,
                priority,
                due_date,
                status_reason,
                COUNT(*) AS count,
                MIN(top_score) AS min_score,
                AVG(top_score) AS avg_score,
                MIN(created_at) AS first_seen_at,
                MAX(created_at) AS last_seen_at
            FROM scoped
            GROUP BY cluster_key, kb_id, kb_key, status, owner_user_id, priority, due_date, status_reason
        )
        SELECT
            grouped.*,
            (
                SELECT query
                FROM scoped latest
                WHERE latest.cluster_key = grouped.cluster_key
                  AND latest.status = grouped.status
                  AND ((latest.kb_id IS NULL AND grouped.kb_id IS NULL) OR latest.kb_id = grouped.kb_id)
                ORDER BY latest.created_at DESC, latest.id DESC
                LIMIT 1
            ) AS representative_query,
            (
                SELECT GROUP_CONCAT(query, char(10))
                FROM (
                    SELECT DISTINCT sample.query
                    FROM scoped sample
                    WHERE sample.cluster_key = grouped.cluster_key
                      AND sample.status = grouped.status
                      AND ((sample.kb_id IS NULL AND grouped.kb_id IS NULL) OR sample.kb_id = grouped.kb_id)
                    ORDER BY sample.created_at DESC, sample.id DESC
                    LIMIT 5
                )
            ) AS sample_queries
        FROM grouped
        ORDER BY count DESC, last_seen_at DESC
        LIMIT ?
        """,
        (*params, max(1, min(int(limit), 100))),
    )
    repeat_count = max(1, int(settings.knowledge_gap_alert_repeat_count))
    items = []
    for row in rows:
        count = int(row.get("count") or 0)
        items.append(
            {
                "cluster_key": row["cluster_key"],
                "representative_query": row.get("representative_query") or "",
                "count": count,
                "kb_id": row.get("kb_id"),
                "kb_key": row.get("kb_key"),
                "mode": None,
                "min_score": round(float(row["min_score"]), 4) if row.get("min_score") is not None else None,
                "avg_score": round(float(row["avg_score"]), 4) if row.get("avg_score") is not None else None,
                "first_seen_at": row["first_seen_at"],
                "last_seen_at": row["last_seen_at"],
                "status": row.get("status") or "open",
                "owner_user_id": row.get("owner_user_id"),
                "priority": _normalize_priority(row.get("priority")),
                "due_date": row.get("due_date"),
                "overdue": bool(row.get("due_date") and (_parse_dt(row.get("due_date")) or datetime.max.replace(tzinfo=timezone.utc)) < datetime.now(timezone.utc)),
                "status_reason": row.get("status_reason"),
                "suggested_action": "create_faq_entry" if count >= repeat_count else None,
                "sample_queries": _sample_queries(row.get("sample_queries")),
            }
        )
    return {"total": len(items), "period_days": int(days), "kb_id": kb_id, "items": items}


def update_gap_cluster_status(
    *,
    cluster_key: str,
    status: str | None = None,
    kb_id: int | None = None,
    owner_user_id: str | None = None,
    priority: str | None = None,
    due_date: str | None = None,
    status_reason: str | None = None,
) -> dict[str, Any]:
    now = utcnow_iso()
    assignments = ["updated_at = ?"]
    params: list[Any] = [now]
    normalized_status = _normalize_status(status) if status else None
    if normalized_status:
        assignments.append("status = ?")
        params.append(normalized_status)
    if owner_user_id is not None:
        assignments.append("owner_user_id = ?")
        params.append(owner_user_id or None)
    if priority is not None:
        assignments.append("priority = ?")
        params.append(_normalize_priority(priority))
    if due_date is not None:
        assignments.append("due_date = ?")
        params.append(due_date or None)
    if status_reason is not None:
        assignments.append("status_reason = ?")
        params.append(status_reason or None)
    set_sql = ", ".join(assignments)
    if kb_id is None:
        updated = execute_sync(
            f"UPDATE knowledge_gaps SET {set_sql} WHERE cluster_key = ?",
            (*params, cluster_key),
        )
    else:
        updated = execute_sync(
            f"UPDATE knowledge_gaps SET {set_sql} WHERE cluster_key = ? AND kb_id = ?",
            (*params, cluster_key, int(kb_id)),
        )
    return {
        "cluster_key": cluster_key,
        "status": normalized_status,
        "kb_id": kb_id,
        "owner_user_id": owner_user_id,
        "priority": _normalize_priority(priority) if priority else None,
        "due_date": due_date,
        "updated": updated,
    }


def _get_gap_cluster(*, cluster_key: str, kb_id: int | None = None, status: str = "new") -> dict[str, Any] | None:
    clauses = ["cluster_key = ?"]
    params: list[Any] = [cluster_key]
    if kb_id is not None:
        clauses.append("kb_id = ?")
        params.append(int(kb_id))
    normalized_status = _normalize_status(status, default="new") if status else ""
    if normalized_status:
        clauses.append("status = ?")
        params.append(normalized_status)
    where_sql = " AND ".join(clauses)
    row = fetch_one_sync(
        f"""
        SELECT
            cluster_key,
            kb_id,
            kb_key,
            status,
            owner_user_id,
            priority,
            due_date,
            status_reason,
            COUNT(*) AS count,
            MIN(top_score) AS min_score,
            AVG(top_score) AS avg_score,
            MIN(created_at) AS first_seen_at,
            MAX(created_at) AS last_seen_at
        FROM knowledge_gaps
        WHERE {where_sql}
        GROUP BY cluster_key, kb_id, kb_key, status, owner_user_id, priority, due_date, status_reason
        """,
        tuple(params),
    )
    if not row:
        return None
    samples = fetch_all_sync(
        f"""
        SELECT DISTINCT query
        FROM knowledge_gaps
        WHERE {where_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT 5
        """,
        tuple(params),
    )
    sample_queries = [str(item.get("query") or "").strip() for item in samples if str(item.get("query") or "").strip()]
    return {
        "cluster_key": row["cluster_key"],
        "kb_id": row.get("kb_id"),
        "kb_key": row.get("kb_key"),
        "status": row.get("status") or normalized_status,
        "owner_user_id": row.get("owner_user_id"),
        "priority": _normalize_priority(row.get("priority")),
        "due_date": row.get("due_date"),
        "status_reason": row.get("status_reason"),
        "count": int(row.get("count") or 0),
        "representative_query": sample_queries[0] if sample_queries else "",
        "sample_queries": sample_queries,
        "min_score": round(float(row["min_score"]), 4) if row.get("min_score") is not None else None,
        "avg_score": round(float(row["avg_score"]), 4) if row.get("avg_score") is not None else None,
        "first_seen_at": row["first_seen_at"],
        "last_seen_at": row["last_seen_at"],
    }


def suggest_faq_pending_action(
    *,
    cluster_key: str,
    kb_id: int | None = None,
    context: RequestContext,
) -> dict[str, Any]:
    cluster = None
    for status in ("new", "triaged", "source_needed", "patch_pending"):
        cluster = _get_gap_cluster(cluster_key=cluster_key, kb_id=kb_id, status=status)
        if cluster:
            break
    if not cluster:
        raise ValueError("Knowledge gap cluster not found")

    target_kb_id = cluster.get("kb_id")
    existing = fetch_one_sync(
        """
        SELECT * FROM pending_actions
        WHERE action_type = 'create_faq_entry'
          AND status IN ('draft', 'approved')
          AND json_extract(payload_json, '$.cluster_key') = ?
          AND ((kb_id IS NULL AND ? IS NULL) OR kb_id = ?)
        ORDER BY id DESC
        LIMIT 1
        """,
        (cluster_key, target_kb_id, target_kb_id),
    )
    if existing:
        return {
            "created": False,
            "pending_action": _serialize_pending_action(existing),
            "cluster": cluster,
        }

    action_context = RequestContext(
        request_id=context.request_id,
        kb_id=target_kb_id,
        kb_key=cluster.get("kb_key"),
        auth=context.auth,
    )
    answer_template = (
        "Draft the approved answer here using verified KB/source material. "
        "Do not publish until the content owner confirms the policy details."
    )
    from app.pending_actions import draft_knowledge_gap_faq_action

    action = draft_knowledge_gap_faq_action(
        cluster_key=cluster_key,
        question=cluster["representative_query"],
        sample_queries=cluster["sample_queries"],
        answer_template=answer_template,
        gap_count=int(cluster["count"]),
        context=action_context,
    )
    update_gap_cluster_status(
        cluster_key=cluster_key,
        status="patch_pending",
        kb_id=target_kb_id,
        status_reason="FAQ/source patch pending admin approval.",
    )
    cluster["status"] = "patch_pending"
    return {"created": True, "pending_action": action, "cluster": cluster}


def create_feedback_gap_if_needed(
    *,
    chat_log: dict[str, Any],
    rating: str,
    reason_code: str | None,
    comment: str | None,
    context: RequestContext,
) -> int | None:
    if str(rating or "").lower() != "down":
        return None
    chat_log_id = int(chat_log["id"])
    existing = fetch_one_sync(
        """
        SELECT id FROM knowledge_gaps
        WHERE chat_log_id = ? AND mode = 'feedback_down'
        LIMIT 1
        """,
        (chat_log_id,),
    )
    if existing:
        return int(existing["id"])
    return record_knowledge_gap(
        chat_log_id=chat_log_id,
        query=str(chat_log.get("user_message") or ""),
        mode="feedback_down",
        top_score=chat_log.get("top_score"),
        session_id=chat_log.get("session_id"),
        context={
            "kb_id": chat_log.get("kb_id"),
            "kb_key": chat_log.get("kb_key"),
            "auth": context.auth.model_dump(),
            "priority": "P1" if reason_code in {"wrong_source", "hallucination", "outdated"} else "P2",
            "status_reason": "Negative chat feedback"
            + (f": {reason_code}" if reason_code else "")
            + (f" - {comment[:180]}" if comment else ""),
        },
    )


def build_knowledge_quality_debt(*, kb_id: int | None = None, days: int = 30) -> dict[str, Any]:
    params: list[Any] = [f"-{max(1, int(days))} days"]
    clauses = ["datetime(created_at) >= datetime('now', ?)"]
    if kb_id is not None:
        clauses.append("kb_id = ?")
        params.append(int(kb_id))
    where = " AND ".join(clauses)
    status_rows = fetch_all_sync(
        f"""
        SELECT status, COUNT(*) AS count
        FROM knowledge_gaps
        WHERE {where}
        GROUP BY status
        ORDER BY count DESC
        """,
        tuple(params),
    )
    priority_rows = fetch_all_sync(
        f"""
        SELECT priority, COUNT(*) AS count
        FROM knowledge_gaps
        WHERE {where}
          AND status IN ('new', 'triaged', 'source_needed', 'patch_pending')
        GROUP BY priority
        ORDER BY priority ASC
        """,
        tuple(params),
    )
    due_rows = fetch_one_sync(
        f"""
        SELECT
            COUNT(*) AS active_count,
            COALESCE(SUM(CASE WHEN due_date IS NOT NULL AND date(due_date) < date('now') THEN 1 ELSE 0 END), 0) AS overdue_count,
            COALESCE(SUM(CASE WHEN due_date IS NOT NULL AND date(due_date) BETWEEN date('now') AND date('now', '+3 days') THEN 1 ELSE 0 END), 0) AS due_soon_count
        FROM knowledge_gaps
        WHERE {where}
          AND status IN ('new', 'triaged', 'source_needed', 'patch_pending')
        """,
        tuple(params),
    ) or {}
    kb_clause = "WHERE kf.kb_id = ?" if kb_id is not None else ""
    kb_params: tuple[Any, ...] = (int(kb_id),) if kb_id is not None else ()
    doc_rows = fetch_one_sync(
        f"""
        SELECT
            COALESCE(SUM(CASE WHEN kf.stale_reason IS NOT NULL THEN 1 ELSE 0 END), 0) AS stale_document_count,
            COALESCE(SUM(CASE WHEN kf.status = 'failed' OR uf.status = 'failed' THEN 1 ELSE 0 END), 0) AS failed_ingest_count,
            COALESCE(SUM(CASE WHEN kf.chunk_count = 0 THEN 1 ELSE 0 END), 0) AS zero_chunk_count
        FROM kb_files kf
        JOIN uploaded_files uf ON uf.id = kf.file_id
        {kb_clause}
        """,
        kb_params,
    ) or {}
    action_rows = fetch_one_sync(
        """
        SELECT COUNT(*) AS count
        FROM pending_actions
        WHERE action_type = 'create_faq_entry'
          AND status IN ('draft', 'approved')
          AND ((kb_id IS NULL AND ? IS NULL) OR kb_id = ?)
        """,
        (kb_id, kb_id),
    ) or {}
    return {
        "period_days": int(days),
        "kb_id": kb_id,
        "active_gap_count": int(due_rows.get("active_count") or 0),
        "overdue_gap_count": int(due_rows.get("overdue_count") or 0),
        "due_soon_gap_count": int(due_rows.get("due_soon_count") or 0),
        "pending_patch_action_count": int(action_rows.get("count") or 0),
        "stale_document_count": int(doc_rows.get("stale_document_count") or 0),
        "failed_ingest_count": int(doc_rows.get("failed_ingest_count") or 0),
        "zero_chunk_count": int(doc_rows.get("zero_chunk_count") or 0),
        "by_status": [
            {"key": row.get("status") or "unknown", "count": int(row.get("count") or 0)}
            for row in status_rows
        ],
        "by_priority": [
            {"key": _normalize_priority(row.get("priority")), "count": int(row.get("count") or 0)}
            for row in priority_rows
        ],
    }


def _serialize_pending_action(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    result: dict[str, Any] | None = None
    try:
        payload = json.loads(row.get("payload_json") or "{}")
    except Exception:
        payload = {}
    try:
        result = json.loads(row.get("result_json") or "null")
    except Exception:
        result = None
    return {
        "id": int(row["id"]),
        "action_type": row["action_type"],
        "risk_level": row["risk_level"],
        "status": row["status"],
        "title": row["title"],
        "summary": row.get("summary") or "",
        "payload": payload if isinstance(payload, dict) else {},
        "result": result if isinstance(result, dict) else None,
        "error_message": row.get("error_message"),
        "created_by_user_id": row.get("created_by_user_id"),
        "approved_by_user_id": row.get("approved_by_user_id"),
        "executed_by_user_id": row.get("executed_by_user_id"),
        "tenant_id": row.get("tenant_id"),
        "org_id": row.get("org_id"),
        "kb_id": row.get("kb_id"),
        "kb_key": row.get("kb_key"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "approved_at": row.get("approved_at"),
        "executed_at": row.get("executed_at"),
        "expires_at": row.get("expires_at"),
    }


def create_knowledge_gap_report(
    *,
    days: int = 7,
    kb_id: int | None = None,
    status: str = "new",
    limit: int = 20,
    context: RequestContext | None = None,
) -> dict[str, Any]:
    period_days = max(1, int(days))
    normalized_status = _normalize_status(status, default="new")
    clusters = list_knowledge_gap_clusters(
        days=period_days,
        kb_id=kb_id,
        status=normalized_status,
        limit=limit,
    )

    clauses = ["datetime(created_at) >= datetime('now', ?)"]
    params: list[Any] = [f"-{period_days} days"]
    if kb_id is not None:
        clauses.append("kb_id = ?")
        params.append(int(kb_id))
    if normalized_status:
        clauses.append("status = ?")
        params.append(normalized_status)
    where_sql = " AND ".join(clauses)
    totals = fetch_one_sync(
        f"""
        SELECT
            COUNT(*) AS event_count,
            COUNT(DISTINCT cluster_key || ':' || COALESCE(kb_id, 'global')) AS cluster_count
        FROM knowledge_gaps
        WHERE {where_sql}
        """,
        tuple(params),
    )
    event_count = int((totals or {}).get("event_count") or 0)
    cluster_count = int((totals or {}).get("cluster_count") or 0)
    top_items = clusters["items"]
    top_labels = [
        f"{item['representative_query']} ({item['count']})"
        for item in top_items[:5]
        if item.get("representative_query")
    ]
    message = (
        f"{period_days}-day knowledge gap report: {event_count} unanswered or low-confidence "
        f"question(s) across {cluster_count} cluster(s)."
    )
    quality_debt = build_knowledge_quality_debt(kb_id=kb_id, days=period_days)
    debt_bits = []
    if quality_debt["overdue_gap_count"]:
        debt_bits.append(f"{quality_debt['overdue_gap_count']} overdue gap(s)")
    if quality_debt["stale_document_count"]:
        debt_bits.append(f"{quality_debt['stale_document_count']} stale doc(s)")
    if quality_debt["failed_ingest_count"]:
        debt_bits.append(f"{quality_debt['failed_ingest_count']} failed ingest(s)")
    if debt_bits:
        message += " Quality debt: " + ", ".join(debt_bits) + "."
    if top_labels:
        message += " Top clusters: " + "; ".join(top_labels)

    notification = None
    try:
        from app.notifications import create_notification

        notification = create_notification(
            event_type="knowledge_gap.weekly_report",
            severity="warning" if event_count else "info",
            title="Knowledge gap report",
            message=message,
            entity_type="knowledge_gap_report",
            entity_id=f"{kb_id or 'global'}:{period_days}:{normalized_status or 'all'}",
            payload={
                "period_days": period_days,
                "kb_id": kb_id,
                "status": normalized_status,
                "event_count": event_count,
                "cluster_count": cluster_count,
                "top_clusters": top_items,
                "quality_debt": quality_debt,
            },
            context=context,
        )
    except Exception:
        notification = None

    return {
        "period_days": period_days,
        "kb_id": kb_id,
        "status": normalized_status,
        "event_count": event_count,
        "cluster_count": cluster_count,
        "top_clusters": top_items,
        "quality_debt": quality_debt,
        "notification": notification,
    }
