"""
Rule-based agent evaluation center for chat answers.

V1 intentionally scores persisted chat logs from operational signals that already
exist in the system. This gives admins a repeatable quality gate without adding
another LLM dependency to the evaluation path.
"""

from __future__ import annotations

import json
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
    ListAgentEvalRunsOutput,
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
        chat_log_id=int(row["chat_log_id"]),
        request_id=row.get("request_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        mode=row.get("mode"),
        top_score=row.get("top_score"),
        feedback_rating=row.get("feedback_rating"),
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


def create_agent_eval_run(payload: CreateAgentEvalRunInput, *, auth: AuthContext) -> dict[str, Any]:
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
