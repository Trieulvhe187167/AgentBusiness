"""
Rule-based agent evaluation center for chat answers.

V1 intentionally scores persisted chat logs from operational signals that already
exist in the system. This gives admins a repeatable quality gate without adding
another LLM dependency to the evaluation path.
"""

from __future__ import annotations

import csv
import json
import re
import uuid
from io import StringIO
from typing import Any

from fastapi import HTTPException

from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import (
    AgentEvalCheck,
    AgentEvalResultItem,
    AgentEvalRunDetail,
    AgentEvalRunItem,
    AuthContext,
    CreateAgentEvalRunInput,
    CreateGoldenDatasetItemInput,
    GoldenDatasetItem,
    GoldenDatasetUploadOutput,
    ListAgentEvalRunsOutput,
    ListGoldenDatasetOutput,
    RequestContext,
)
from app.observability import trace_span


def _parse_json(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _serialize_run(row: dict[str, Any]) -> AgentEvalRunItem:
    return AgentEvalRunItem(
        id=int(row["id"]),
        name=row["name"],
        status=row["status"],
        source=row.get("source") or "chat_logs",
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        period_days=int(row.get("period_days") or 0),
        sample_size=int(row.get("sample_size") or 0),
        pass_count=int(row.get("pass_count") or 0),
        warn_count=int(row.get("warn_count") or 0),
        fail_count=int(row.get("fail_count") or 0),
        avg_score=round(float(row["avg_score"]), 2) if row.get("avg_score") is not None else None,
        created_by_user_id=row.get("created_by_user_id"),
        created_at=row["created_at"],
        completed_at=row.get("completed_at"),
    )


def _serialize_result(row: dict[str, Any]) -> AgentEvalResultItem:
    checks = [
        AgentEvalCheck(
            name=str(item.get("name") or "unknown"),
            status=str(item.get("status") or "warn"),
            impact=int(item.get("impact") or 0),
            message=str(item.get("message") or ""),
        )
        for item in _parse_json(row.get("checks_json"), [])
        if isinstance(item, dict)
    ]
    return AgentEvalResultItem(
        id=int(row["id"]),
        run_id=int(row["run_id"]),
        chat_log_id=int(row["chat_log_id"]) if row.get("chat_log_id") is not None else None,
        golden_item_id=row.get("golden_item_id"),
        request_id=row.get("request_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        mode=row.get("mode"),
        top_score=row.get("top_score"),
        feedback_rating=row.get("feedback_rating"),
        expected_answer=row.get("expected_answer"),
        answer_similarity=row.get("answer_similarity"),
        recall_at_k=row.get("recall_at_k"),
        citation_accuracy=row.get("citation_accuracy"),
        verdict=row["verdict"],
        score=round(float(row.get("score") or 0), 2),
        checks=checks,
        reason=row.get("reason"),
        user_message=row.get("user_message") or "",
        answer_text=row.get("answer_text") or "",
        created_at=row["created_at"],
    )


def _check(name: str, status: str, impact: int, message: str) -> dict[str, Any]:
    return {"name": name, "status": status, "impact": int(impact), "message": message}


def _parse_list(raw: str | None) -> list[str]:
    parsed = _parse_json(raw, [])
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _serialize_golden_item(row: dict[str, Any]) -> GoldenDatasetItem:
    return GoldenDatasetItem(
        id=int(row["id"]),
        kb_id=int(row["kb_id"]),
        question=row["question"],
        expected_answer=row["expected_answer"],
        expected_source_file_id=row.get("expected_source_file_id"),
        expected_keywords=_parse_list(row.get("expected_keywords_json")),
        tags=_parse_list(row.get("tags_json")),
        active=bool(row.get("active")),
        created_by_user_id=row.get("created_by_user_id"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _citation_count(raw: str | None) -> int:
    citations = _parse_json(raw, [])
    if isinstance(citations, list):
        return len(citations)
    return 0


def _latest_feedback_rating(row: dict[str, Any]) -> str | None:
    down = int(row.get("feedback_down") or 0)
    up = int(row.get("feedback_up") or 0)
    if down > 0:
        return "down"
    if up > 0:
        return "up"
    return None


def _score_chat(row: dict[str, Any], *, min_pass_score: int, min_warn_score: int) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    answer = str(row.get("answer_text") or "").strip()
    mode = str(row.get("mode") or "unknown")
    top_score = row.get("top_score")
    citation_count = _citation_count(row.get("citations_json"))
    feedback_rating = _latest_feedback_rating(row)

    if answer:
        impact = 0 if len(answer) >= 25 else -15
        checks.append(
            _check(
                "answer_completeness",
                "pass" if impact == 0 else "warn",
                impact,
                "Answer is present." if impact == 0 else "Answer is very short and may be incomplete.",
            )
        )
    else:
        checks.append(_check("answer_completeness", "fail", -60, "No answer text was returned."))

    if mode == "fallback":
        checks.append(_check("routing_mode", "warn", -30, "Chat used fallback mode."))
    else:
        checks.append(_check("routing_mode", "pass", 0, f"Chat mode is {mode}."))

    if citation_count > 0:
        checks.append(_check("citations", "pass", 0, f"{citation_count} citation(s) attached."))
    else:
        checks.append(_check("citations", "warn", -15, "No citation was attached to the answer."))

    if top_score is None:
        checks.append(_check("retrieval_score", "warn", -10, "No retrieval score was recorded."))
    elif float(top_score) >= 0.60:
        checks.append(_check("retrieval_score", "pass", 0, f"Top score {float(top_score):.2f}."))
    elif float(top_score) >= 0.35:
        checks.append(_check("retrieval_score", "warn", -15, f"Top score {float(top_score):.2f} is borderline."))
    else:
        checks.append(_check("retrieval_score", "fail", -25, f"Top score {float(top_score):.2f} is low."))

    if feedback_rating == "down":
        checks.append(_check("user_feedback", "fail", -45, "User marked this answer as not helpful."))
    elif feedback_rating == "up":
        checks.append(_check("user_feedback", "pass", 5, "User marked this answer as helpful."))
    else:
        checks.append(_check("user_feedback", "warn", -5, "No user feedback recorded."))

    score = max(0, min(100, 100 + sum(int(item["impact"]) for item in checks)))
    if score >= min_pass_score:
        verdict = "pass"
    elif score >= min_warn_score:
        verdict = "warn"
    else:
        verdict = "fail"
    failing = [item["message"] for item in checks if item["status"] in {"fail", "warn"}]
    reason = failing[0] if failing else "All evaluation checks passed."
    return {
        "score": float(score),
        "verdict": verdict,
        "checks": checks,
        "feedback_rating": feedback_rating,
        "reason": reason,
    }


def create_golden_dataset_item(payload: CreateGoldenDatasetItemInput, *, auth: AuthContext) -> dict[str, Any]:
    kb = fetch_one_sync("SELECT id FROM knowledge_bases WHERE id = ?", (int(payload.kb_id),))
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge Base not found")
    if payload.expected_source_file_id is not None:
        file_row = fetch_one_sync("SELECT id FROM uploaded_files WHERE id = ?", (int(payload.expected_source_file_id),))
        if not file_row:
            raise HTTPException(status_code=404, detail="Expected source file not found")

    now = utcnow_iso()
    item_id = int(
        execute_sync(
            """
            INSERT INTO eval_golden_dataset (
                kb_id, question, expected_answer, expected_source_file_id,
                expected_keywords_json, tags_json, active,
                created_by_user_id, tenant_id, org_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(payload.kb_id),
                payload.question,
                payload.expected_answer,
                payload.expected_source_file_id,
                json.dumps(payload.expected_keywords, ensure_ascii=False),
                json.dumps(payload.tags, ensure_ascii=False),
                1 if payload.active else 0,
                auth.user_id,
                auth.tenant_id,
                auth.org_id,
                now,
                now,
            ),
        )
        or 0
    )
    row = fetch_one_sync("SELECT * FROM eval_golden_dataset WHERE id = ?", (item_id,))
    if not row:
        raise RuntimeError("Golden dataset item was not persisted")
    return _serialize_golden_item(row).model_dump()


def list_golden_dataset(*, kb_id: int | None = None, active: bool | None = None, limit: int = 100) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if kb_id is not None:
        clauses.append("kb_id = ?")
        params.append(int(kb_id))
    if active is not None:
        clauses.append("active = ?")
        params.append(1 if active else 0)
    params.append(max(1, min(int(limit), 500)))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = fetch_all_sync(
        f"""
        SELECT *
        FROM eval_golden_dataset
        {where}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        tuple(params),
    )
    items = [_serialize_golden_item(row) for row in rows]
    return ListGoldenDatasetOutput(total=len(items), items=items).model_dump()


def upload_golden_dataset_csv(
    *,
    content: bytes,
    default_kb_id: int | None,
    auth: AuthContext,
) -> dict[str, Any]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    created: list[dict[str, Any]] = []
    for line_no, row in enumerate(reader, start=2):
        question = (row.get("question") or "").strip()
        expected_answer = (row.get("expected_answer") or row.get("answer") or "").strip()
        kb_raw = (row.get("kb_id") or "").strip()
        kb_id = int(kb_raw) if kb_raw else default_kb_id
        if not kb_id:
            raise HTTPException(status_code=400, detail=f"Missing kb_id at CSV line {line_no}")
        if not question or not expected_answer:
            raise HTTPException(status_code=400, detail=f"Missing question or expected_answer at CSV line {line_no}")
        expected_source = (row.get("expected_source_file_id") or "").strip()
        payload = CreateGoldenDatasetItemInput(
            kb_id=int(kb_id),
            question=question,
            expected_answer=expected_answer,
            expected_source_file_id=int(expected_source) if expected_source else None,
            expected_keywords=row.get("expected_keywords") or "",
            tags=row.get("tags") or "",
            active=str(row.get("active") or "true").strip().lower() not in {"0", "false", "no"},
        )
        created.append(create_golden_dataset_item(payload, auth=auth))
    return GoldenDatasetUploadOutput(
        created=len(created),
        items=[GoldenDatasetItem(**item) for item in created],
    ).model_dump()


def _select_eval_candidates(payload: CreateAgentEvalRunInput) -> list[dict[str, Any]]:
    clauses = ["datetime(cl.created_at) >= datetime('now', ?)"]
    params: list[Any] = [f"-{int(payload.days)} days"]
    if payload.kb_id is not None:
        clauses.append("cl.kb_id = ?")
        params.append(int(payload.kb_id))
    params.append(int(payload.limit))
    return fetch_all_sync(
        f"""
        SELECT
            cl.*,
            COALESCE(SUM(CASE WHEN cf.rating = 'up' THEN 1 ELSE 0 END), 0) AS feedback_up,
            COALESCE(SUM(CASE WHEN cf.rating = 'down' THEN 1 ELSE 0 END), 0) AS feedback_down
        FROM chat_logs cl
        LEFT JOIN chat_feedback cf ON cf.chat_log_id = cl.id
        WHERE {' AND '.join(clauses)}
        GROUP BY cl.id
        ORDER BY cl.created_at DESC, cl.id DESC
        LIMIT ?
        """,
        tuple(params),
    )


def _select_golden_candidates(payload: CreateAgentEvalRunInput) -> list[dict[str, Any]]:
    clauses = ["active = 1"]
    params: list[Any] = []
    if payload.kb_id is not None:
        clauses.append("kb_id = ?")
        params.append(int(payload.kb_id))
    params.append(int(payload.limit))
    return fetch_all_sync(
        f"""
        SELECT *
        FROM eval_golden_dataset
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        tuple(params),
    )


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "") if len(token.strip()) > 1}


def _token_f1(actual: str, expected: str) -> float:
    actual_tokens = _tokens(actual)
    expected_tokens = _tokens(expected)
    if not actual_tokens or not expected_tokens:
        return 0.0
    overlap = len(actual_tokens & expected_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)


def _collect_rag_answer(question: str, *, kb_id: int, auth: AuthContext) -> dict[str, Any]:
    from app.rag import rag_stream, retrieve

    request_id = f"eval-{uuid.uuid4().hex[:10]}"
    context = RequestContext(
        request_id=request_id,
        kb_id=kb_id,
        auth=auth,
    )
    retrieved = retrieve(question, kb_id=kb_id, auth_context=context.auth.model_dump())
    answer_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    start_event: dict[str, Any] = {}
    done_event: dict[str, Any] = {}
    for event in rag_stream(
        question,
        session_id=f"eval-{uuid.uuid4().hex[:10]}",
        kb_id=kb_id,
        request_context=context,
    ):
        name = event.get("event")
        data = event.get("data") or {}
        if name == "start":
            start_event = data
        elif name == "token":
            answer_parts.append(str(data.get("text") or ""))
        elif name == "citations":
            raw_items = data.get("items") or []
            citations = raw_items if isinstance(raw_items, list) else []
        elif name == "done":
            done_event = data

    chat = fetch_one_sync("SELECT id FROM chat_logs WHERE request_id = ?", (request_id,))
    if not chat:
        raise RuntimeError("Golden evaluation RAG run did not persist a chat log")

    return {
        "request_id": request_id,
        "chat_log_id": int(chat["id"]),
        "answer_text": "".join(answer_parts).strip(),
        "citations": citations,
        "retrieved": retrieved,
        "mode": start_event.get("mode"),
        "top_score": start_event.get("score"),
        "kb_key": start_event.get("kb_key"),
        "latency_ms": done_event.get("latency_ms"),
    }


def _score_golden_item(row: dict[str, Any], *, auth: AuthContext, min_pass_score: int, min_warn_score: int) -> dict[str, Any]:
    rag_result = _collect_rag_answer(row["question"], kb_id=int(row["kb_id"]), auth=auth)
    expected_answer = str(row.get("expected_answer") or "")
    answer_similarity = _token_f1(rag_result["answer_text"], expected_answer)

    expected_source_file_id = row.get("expected_source_file_id")
    recall_at_k: float | None = None
    if expected_source_file_id is not None:
        expected_source = str(expected_source_file_id)
        recall_at_k = 1.0 if any(str(item.get("source_id") or item.get("file_id")) == expected_source for item in rag_result["retrieved"]) else 0.0

    keywords = _parse_list(row.get("expected_keywords_json"))
    citation_accuracy: float | None = None
    if keywords:
        citation_text = " ".join(
            str(item.get("content_preview") or "") + " " + str(item.get("filename") or "")
            for item in rag_result["citations"]
        ).lower()
        matched = sum(1 for keyword in keywords if keyword.lower() in citation_text or keyword.lower() in rag_result["answer_text"].lower())
        citation_accuracy = round(matched / len(keywords), 4)
    elif expected_source_file_id is not None:
        citation_accuracy = recall_at_k

    weighted: list[tuple[float, float]] = [(answer_similarity, 0.6)]
    if recall_at_k is not None:
        weighted.append((recall_at_k, 0.25))
    if citation_accuracy is not None:
        weighted.append((citation_accuracy, 0.15))
    weight_total = sum(weight for _, weight in weighted) or 1.0
    score = round(sum(value * weight for value, weight in weighted) / weight_total * 100, 2)

    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "answer_similarity",
            "pass" if answer_similarity >= 0.65 else "warn" if answer_similarity >= 0.35 else "fail",
            0,
            f"Answer similarity {answer_similarity:.2f}.",
        )
    )
    if recall_at_k is not None:
        checks.append(
            _check(
                "recall_at_k",
                "pass" if recall_at_k >= 1.0 else "fail",
                0,
                "Expected source was retrieved." if recall_at_k >= 1.0 else "Expected source was not retrieved.",
            )
        )
    if citation_accuracy is not None:
        checks.append(
            _check(
                "citation_accuracy",
                "pass" if citation_accuracy >= 0.8 else "warn" if citation_accuracy >= 0.4 else "fail",
                0,
                f"Citation accuracy {citation_accuracy:.2f}.",
            )
        )
    if rag_result["mode"] == "fallback":
        checks.append(_check("routing_mode", "warn", 0, "RAG used fallback mode."))

    if score >= min_pass_score:
        verdict = "pass"
    elif score >= min_warn_score:
        verdict = "warn"
    else:
        verdict = "fail"
    failing = [item["message"] for item in checks if item["status"] in {"fail", "warn"}]
    return {
        **rag_result,
        "score": score,
        "verdict": verdict,
        "checks": checks,
        "reason": failing[0] if failing else "Golden dataset checks passed.",
        "answer_similarity": answer_similarity,
        "recall_at_k": recall_at_k,
        "citation_accuracy": citation_accuracy,
    }


def _maybe_create_quality_alert(run_id: int, *, kb_id: int | None, avg_score: float | None, threshold: float) -> None:
    if avg_score is None or threshold <= 0:
        return
    params: list[Any] = [int(run_id)]
    kb_clause = ""
    if kb_id is not None:
        kb_clause = "AND kb_id = ?"
        params.append(int(kb_id))
    previous = fetch_one_sync(
        f"""
        SELECT id, avg_score
        FROM agent_eval_runs
        WHERE source = 'golden_dataset'
          AND avg_score IS NOT NULL
          AND id <> ?
          {kb_clause}
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        tuple(params),
    )
    if not previous or previous.get("avg_score") is None:
        return
    previous_score = float(previous["avg_score"])
    drop = previous_score - float(avg_score)
    if drop < threshold:
        return
    try:
        from app.notifications import create_notification

        create_notification(
            event_type="evaluation.quality_drop",
            severity="critical" if drop >= threshold * 2 else "warning",
            title="RAG quality regression detected",
            message=f"Golden eval average score dropped by {drop:.1f} points ({previous_score:.1f} -> {avg_score:.1f}).",
            entity_type="agent_eval_run",
            entity_id=str(run_id),
            payload={"run_id": run_id, "previous_run_id": previous["id"], "previous_avg_score": previous_score, "avg_score": avg_score, "drop": drop},
            context=RequestContext(request_id=f"eval-alert-{run_id}", kb_id=kb_id, auth={"user_id": "evaluation-monitor", "roles": ["admin"], "channel": "scheduler"}),
        )
    except Exception:
        # Evaluation should never fail just because alert delivery failed.
        pass


def _create_golden_eval_run(payload: CreateAgentEvalRunInput, *, auth: AuthContext) -> dict[str, Any]:
    rows = _select_golden_candidates(payload)
    now = utcnow_iso()
    kb_key = None
    if payload.kb_id is not None:
        kb_row = fetch_one_sync("SELECT key FROM knowledge_bases WHERE id = ?", (int(payload.kb_id),))
        kb_key = kb_row.get("key") if kb_row else None
    config = {
        "limit": payload.limit,
        "kb_id": payload.kb_id,
        "min_pass_score": payload.min_pass_score,
        "min_warn_score": payload.min_warn_score,
        "alert_drop_threshold": payload.alert_drop_threshold,
        "scorer": "golden_rule_v1",
        "metrics": ["answer_similarity", "recall_at_k", "citation_accuracy"],
    }
    run_name = payload.name or "Golden dataset regression eval"
    run_id = int(
        execute_sync(
            """
            INSERT INTO agent_eval_runs (
                name, status, source, kb_id, kb_key, period_days, sample_size,
                config_json, created_by_user_id, tenant_id, org_id, created_at, completed_at
            ) VALUES (?, 'completed', 'golden_dataset', ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                payload.kb_id,
                kb_key,
                len(rows),
                json.dumps(config, ensure_ascii=False, sort_keys=True),
                auth.user_id,
                auth.tenant_id,
                auth.org_id,
                now,
                now,
            ),
        )
        or 0
    )

    pass_count = warn_count = fail_count = 0
    total_score = 0.0
    for row in rows:
        scored = _score_golden_item(
            row,
            auth=auth,
            min_pass_score=payload.min_pass_score,
            min_warn_score=payload.min_warn_score,
        )
        verdict = scored["verdict"]
        pass_count += 1 if verdict == "pass" else 0
        warn_count += 1 if verdict == "warn" else 0
        fail_count += 1 if verdict == "fail" else 0
        total_score += float(scored["score"])
        execute_sync(
            """
            INSERT INTO agent_eval_results (
                run_id, chat_log_id, golden_item_id, request_id, kb_id, kb_key, mode, top_score,
                feedback_rating, expected_answer, answer_similarity, recall_at_k, citation_accuracy,
                verdict, score, checks_json, reason, user_message, answer_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(scored["chat_log_id"]),
                int(row["id"]),
                scored["request_id"],
                int(row["kb_id"]),
                scored.get("kb_key"),
                scored.get("mode"),
                scored.get("top_score"),
                row.get("expected_answer"),
                scored.get("answer_similarity"),
                scored.get("recall_at_k"),
                scored.get("citation_accuracy"),
                verdict,
                scored["score"],
                json.dumps(scored["checks"], ensure_ascii=False),
                scored["reason"],
                row.get("question") or "",
                scored["answer_text"],
                now,
            ),
        )

    avg_score = round(total_score / len(rows), 2) if rows else None
    execute_sync(
        """
        UPDATE agent_eval_runs
        SET pass_count = ?, warn_count = ?, fail_count = ?, avg_score = ?
        WHERE id = ?
        """,
        (pass_count, warn_count, fail_count, avg_score, run_id),
    )
    _maybe_create_quality_alert(
        run_id,
        kb_id=payload.kb_id,
        avg_score=avg_score,
        threshold=payload.alert_drop_threshold,
    )
    return get_agent_eval_run(run_id)


def create_agent_eval_run(payload: CreateAgentEvalRunInput, *, auth: AuthContext) -> dict[str, Any]:
    if payload.source == "golden_dataset":
        return _create_golden_eval_run(payload, auth=auth)

    with trace_span(
        "agent.eval_run",
        {
            "agent.eval.days": payload.days,
            "agent.eval.limit": payload.limit,
            "agent.eval.kb_id": payload.kb_id,
            "agent.eval.scorer": "rule_based_v1",
        },
    ) as span:
        rows = _select_eval_candidates(payload)
        span.set_attribute("agent.eval.sample_size", len(rows))
        now = utcnow_iso()
        kb_key = rows[0].get("kb_key") if payload.kb_id is not None and rows else None
        config = {
            "days": payload.days,
            "limit": payload.limit,
            "kb_id": payload.kb_id,
            "min_pass_score": payload.min_pass_score,
            "min_warn_score": payload.min_warn_score,
            "scorer": "rule_based_v1",
        }
        run_name = payload.name or f"Chat quality eval - last {payload.days} day(s)"
        run_id = int(
            execute_sync(
                """
                INSERT INTO agent_eval_runs (
                    name, status, source, kb_id, kb_key, period_days, sample_size,
                    config_json, created_by_user_id, tenant_id, org_id, created_at, completed_at
                ) VALUES (?, 'completed', 'chat_logs', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_name,
                    payload.kb_id,
                    kb_key,
                    payload.days,
                    len(rows),
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                    auth.user_id,
                    auth.tenant_id,
                    auth.org_id,
                    now,
                    now,
                ),
            )
            or 0
        )

        pass_count = warn_count = fail_count = 0
        total_score = 0.0
        for row in rows:
            scored = _score_chat(
                row,
                min_pass_score=payload.min_pass_score,
                min_warn_score=payload.min_warn_score,
            )
            verdict = scored["verdict"]
            pass_count += 1 if verdict == "pass" else 0
            warn_count += 1 if verdict == "warn" else 0
            fail_count += 1 if verdict == "fail" else 0
            total_score += float(scored["score"])
            execute_sync(
                """
                INSERT INTO agent_eval_results (
                    run_id, chat_log_id, request_id, kb_id, kb_key, mode, top_score,
                    feedback_rating, verdict, score, checks_json, reason,
                    user_message, answer_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(row["id"]),
                    row.get("request_id"),
                    row.get("kb_id"),
                    row.get("kb_key"),
                    row.get("mode"),
                    row.get("top_score"),
                    scored["feedback_rating"],
                    verdict,
                    scored["score"],
                    json.dumps(scored["checks"], ensure_ascii=False),
                    scored["reason"],
                    row.get("user_message") or "",
                    row.get("answer_text") or "",
                    now,
                ),
            )

        avg_score = round(total_score / len(rows), 2) if rows else None
        execute_sync(
            """
            UPDATE agent_eval_runs
            SET pass_count = ?, warn_count = ?, fail_count = ?, avg_score = ?
            WHERE id = ?
            """,
            (pass_count, warn_count, fail_count, avg_score, run_id),
        )
        span.set_attribute("agent.eval.run_id", run_id)
        span.set_attribute("agent.eval.pass_count", pass_count)
        span.set_attribute("agent.eval.warn_count", warn_count)
        span.set_attribute("agent.eval.fail_count", fail_count)
        if avg_score is not None:
            span.set_attribute("agent.eval.avg_score", avg_score)
        return get_agent_eval_run(run_id)


def list_agent_eval_runs(*, limit: int = 20) -> dict[str, Any]:
    rows = fetch_all_sync(
        """
        SELECT *
        FROM agent_eval_runs
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (max(1, min(int(limit), 100)),),
    )
    items = [_serialize_run(row) for row in rows]
    return ListAgentEvalRunsOutput(total=len(items), items=items).model_dump()


def get_agent_eval_run(run_id: int, *, limit: int = 100) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM agent_eval_runs WHERE id = ?", (int(run_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Agent eval run not found")
    results = fetch_all_sync(
        """
        SELECT *
        FROM agent_eval_results
        WHERE run_id = ?
        ORDER BY
            CASE verdict WHEN 'fail' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END,
            score ASC,
            id ASC
        LIMIT ?
        """,
        (int(run_id), max(1, min(int(limit), 500))),
    )
    base = _serialize_run(row)
    return AgentEvalRunDetail(
        **base.model_dump(),
        config=_parse_json(row.get("config_json"), {}),
        results=[_serialize_result(result) for result in results],
    ).model_dump()
