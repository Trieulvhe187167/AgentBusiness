"""
Operational analytics for the admin dashboard.
"""

from __future__ import annotations

from app.database import fetch_all, fetch_one
from app.models import (
    AnalyticsBreakdownItem,
    AnalyticsDashboardOutput,
    AnalyticsSummary,
    AnalyticsTimeBucket,
)


def _period_modifier(days: int) -> str:
    return f"-{int(days)} days"


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _kb_clause(alias: str, kb_id: int | None) -> tuple[str, tuple[int, ...]]:
    if kb_id is None:
        return "", ()
    return f" AND {alias}.kb_id = ?", (int(kb_id),)


def _row_int(row: dict | None, key: str) -> int:
    return int((row or {}).get(key) or 0)


def _row_float(row: dict | None, key: str) -> float | None:
    value = (row or {}).get(key)
    return round(float(value), 2) if value is not None else None


def _breakdown(rows: list[dict], key_field: str = "key", count_field: str = "count") -> list[AnalyticsBreakdownItem]:
    return [
        AnalyticsBreakdownItem(
            key=str(row.get(key_field) or "unknown"),
            count=int(row.get(count_field) or 0),
        )
        for row in rows
    ]


async def build_analytics_dashboard(days: int = 7, kb_id: int | None = None) -> AnalyticsDashboardOutput:
    period = _period_modifier(days)
    chat_kb_sql, chat_kb_params = _kb_clause("cl", kb_id)
    tool_kb_sql, tool_kb_params = _kb_clause("tal", kb_id)
    job_kb_sql, job_kb_params = _kb_clause("bj", kb_id)
    ticket_kb_sql, ticket_kb_params = _kb_clause("st", kb_id)
    pending_kb_sql, pending_kb_params = _kb_clause("pa", kb_id)
    file_kb_sql, file_kb_params = _kb_clause("kf", kb_id)

    chat_row = await fetch_one(
        f"""
        SELECT
            COUNT(*) AS chat_count,
            COUNT(DISTINCT NULLIF(user_id, '')) AS unique_users,
            AVG(latency_ms) AS avg_latency_ms,
            SUM(CASE WHEN mode = 'fallback' THEN 1 ELSE 0 END) AS fallback_count,
            SUM(COALESCE(llm_input_tokens, 0)) AS llm_input_tokens,
            SUM(COALESCE(llm_output_tokens, 0)) AS llm_output_tokens,
            SUM(COALESCE(llm_total_tokens, 0)) AS llm_total_tokens,
            SUM(COALESCE(llm_cached_tokens, 0)) AS llm_cached_tokens
        FROM chat_logs cl
        WHERE datetime(cl.created_at) >= datetime('now', ?)
        {chat_kb_sql}
        """,
        (period, *chat_kb_params),
    )
    feedback_row = await fetch_one(
        f"""
        SELECT
            COUNT(cf.id) AS feedback_total,
            SUM(CASE WHEN cf.rating = 'up' THEN 1 ELSE 0 END) AS feedback_up,
            SUM(CASE WHEN cf.rating = 'down' THEN 1 ELSE 0 END) AS feedback_down
        FROM chat_feedback cf
        JOIN chat_logs cl ON cl.id = cf.chat_log_id
        WHERE datetime(cf.created_at) >= datetime('now', ?)
        {chat_kb_sql}
        """,
        (period, *chat_kb_params),
    )
    tool_row = await fetch_one(
        f"""
        SELECT
            COUNT(*) AS tool_calls,
            SUM(CASE WHEN tool_status NOT IN ('success', 'clarify') THEN 1 ELSE 0 END) AS tool_errors
        FROM tool_audit_logs tal
        WHERE datetime(tal.created_at) >= datetime('now', ?)
        {tool_kb_sql}
        """,
        (period, *tool_kb_params),
    )
    job_row = await fetch_one(
        f"""
        SELECT
            COUNT(*) AS background_jobs_total,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS background_jobs_failed
        FROM background_jobs bj
        WHERE datetime(bj.created_at) >= datetime('now', ?)
        {job_kb_sql}
        """,
        (period, *job_kb_params),
    )
    pending_row = await fetch_one(
        f"""
        SELECT COUNT(*) AS pending_actions_open
        FROM pending_actions pa
        WHERE pa.status IN ('draft', 'approved')
        {pending_kb_sql}
        """,
        pending_kb_params,
    )
    ticket_row = await fetch_one(
        f"""
        SELECT
            SUM(CASE WHEN COALESCE(workflow_status, status) NOT IN ('resolved', 'closed') THEN 1 ELSE 0 END) AS support_tickets_open,
            SUM(CASE WHEN COALESCE(workflow_status, status) = 'escalated' THEN 1 ELSE 0 END) AS support_tickets_escalated,
            SUM(CASE WHEN sla_breached_at IS NOT NULL THEN 1 ELSE 0 END) AS sla_overdue
        FROM support_tickets st
        WHERE datetime(st.created_at) >= datetime('now', ?)
        {ticket_kb_sql}
        """,
        (period, *ticket_kb_params),
    )
    file_row = await fetch_one(
        f"""
        SELECT
            COUNT(DISTINCT uf.id) AS uploaded_files,
            COUNT(DISTINCT CASE WHEN uf.status = 'ingested' OR kf.status = 'ingested' THEN uf.id END) AS ingested_files
        FROM uploaded_files uf
        LEFT JOIN kb_files kf ON kf.file_id = uf.id
        WHERE datetime(uf.created_at) >= datetime('now', ?)
        {file_kb_sql}
        """,
        (period, *file_kb_params),
    )

    feedback_total = _row_int(feedback_row, "feedback_total")
    feedback_up = _row_int(feedback_row, "feedback_up")
    tool_calls = _row_int(tool_row, "tool_calls")
    tool_errors = _row_int(tool_row, "tool_errors")
    llm_input_tokens = _row_int(chat_row, "llm_input_tokens")
    llm_cached_tokens = _row_int(chat_row, "llm_cached_tokens")
    summary = AnalyticsSummary(
        chat_count=_row_int(chat_row, "chat_count"),
        unique_users=_row_int(chat_row, "unique_users"),
        avg_latency_ms=_row_float(chat_row, "avg_latency_ms"),
        fallback_count=_row_int(chat_row, "fallback_count"),
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=_row_int(chat_row, "llm_output_tokens"),
        llm_total_tokens=_row_int(chat_row, "llm_total_tokens"),
        llm_cached_tokens=llm_cached_tokens,
        llm_cached_input_rate=_rate(llm_cached_tokens, llm_input_tokens),
        feedback_total=feedback_total,
        feedback_up=feedback_up,
        feedback_down=_row_int(feedback_row, "feedback_down"),
        positive_rate=_rate(feedback_up, feedback_total),
        tool_calls=tool_calls,
        tool_error_rate=_rate(tool_errors, tool_calls),
        background_jobs_total=_row_int(job_row, "background_jobs_total"),
        background_jobs_failed=_row_int(job_row, "background_jobs_failed"),
        pending_actions_open=_row_int(pending_row, "pending_actions_open"),
        support_tickets_open=_row_int(ticket_row, "support_tickets_open"),
        support_tickets_escalated=_row_int(ticket_row, "support_tickets_escalated"),
        sla_overdue=_row_int(ticket_row, "sla_overdue"),
        uploaded_files=_row_int(file_row, "uploaded_files"),
        ingested_files=_row_int(file_row, "ingested_files"),
    )

    timeseries_rows = await fetch_all(
        f"""
        WITH RECURSIVE days AS (
            SELECT date('now', ?) AS bucket
            UNION ALL
            SELECT date(bucket, '+1 day') FROM days WHERE bucket < date('now')
        ),
        chat_counts AS (
            SELECT date(cl.created_at) AS bucket, COUNT(*) AS chats
            FROM chat_logs cl
            WHERE datetime(cl.created_at) >= datetime('now', ?)
            {chat_kb_sql}
            GROUP BY date(cl.created_at)
        ),
        feedback_counts AS (
            SELECT
                date(cf.created_at) AS bucket,
                SUM(CASE WHEN cf.rating = 'up' THEN 1 ELSE 0 END) AS feedback_up,
                SUM(CASE WHEN cf.rating = 'down' THEN 1 ELSE 0 END) AS feedback_down
            FROM chat_feedback cf
            JOIN chat_logs cl ON cl.id = cf.chat_log_id
            WHERE datetime(cf.created_at) >= datetime('now', ?)
            {chat_kb_sql}
            GROUP BY date(cf.created_at)
        ),
        tool_counts AS (
            SELECT date(tal.created_at) AS bucket, COUNT(*) AS tool_calls
            FROM tool_audit_logs tal
            WHERE datetime(tal.created_at) >= datetime('now', ?)
            {tool_kb_sql}
            GROUP BY date(tal.created_at)
        ),
        job_counts AS (
            SELECT date(bj.created_at) AS bucket, COUNT(*) AS job_failures
            FROM background_jobs bj
            WHERE datetime(bj.created_at) >= datetime('now', ?)
              AND bj.status = 'failed'
            {job_kb_sql}
            GROUP BY date(bj.created_at)
        ),
        ticket_counts AS (
            SELECT date(st.created_at) AS bucket, COUNT(*) AS support_tickets
            FROM support_tickets st
            WHERE datetime(st.created_at) >= datetime('now', ?)
            {ticket_kb_sql}
            GROUP BY date(st.created_at)
        )
        SELECT
            days.bucket,
            COALESCE(chat_counts.chats, 0) AS chats,
            COALESCE(feedback_counts.feedback_up, 0) AS feedback_up,
            COALESCE(feedback_counts.feedback_down, 0) AS feedback_down,
            COALESCE(tool_counts.tool_calls, 0) AS tool_calls,
            COALESCE(job_counts.job_failures, 0) AS job_failures,
            COALESCE(ticket_counts.support_tickets, 0) AS support_tickets
        FROM days
        LEFT JOIN chat_counts ON chat_counts.bucket = days.bucket
        LEFT JOIN feedback_counts ON feedback_counts.bucket = days.bucket
        LEFT JOIN tool_counts ON tool_counts.bucket = days.bucket
        LEFT JOIN job_counts ON job_counts.bucket = days.bucket
        LEFT JOIN ticket_counts ON ticket_counts.bucket = days.bucket
        ORDER BY days.bucket ASC
        """,
        (
            f"-{max(int(days) - 1, 0)} days",
            period,
            *chat_kb_params,
            period,
            *chat_kb_params,
            period,
            *tool_kb_params,
            period,
            *job_kb_params,
            period,
            *ticket_kb_params,
        ),
    )
    timeseries = [
        AnalyticsTimeBucket(
            bucket=row["bucket"],
            chats=int(row.get("chats") or 0),
            feedback_up=int(row.get("feedback_up") or 0),
            feedback_down=int(row.get("feedback_down") or 0),
            tool_calls=int(row.get("tool_calls") or 0),
            job_failures=int(row.get("job_failures") or 0),
            support_tickets=int(row.get("support_tickets") or 0),
        )
        for row in timeseries_rows
    ]

    chat_modes = _breakdown(
        await fetch_all(
            f"""
            SELECT COALESCE(mode, 'unknown') AS key, COUNT(*) AS count
            FROM chat_logs cl
            WHERE datetime(cl.created_at) >= datetime('now', ?)
            {chat_kb_sql}
            GROUP BY COALESCE(mode, 'unknown')
            ORDER BY count DESC
            LIMIT 8
            """,
            (period, *chat_kb_params),
        )
    )
    kb_usage_rows = await fetch_all(
        f"""
        SELECT COALESCE(cl.kb_key, 'global') AS key, COUNT(*) AS count
        FROM chat_logs cl
        WHERE datetime(cl.created_at) >= datetime('now', ?)
        {chat_kb_sql}
        GROUP BY COALESCE(cl.kb_key, 'global')
        ORDER BY count DESC
        LIMIT 8
        """,
        (period, *chat_kb_params),
    )
    top_tools = _breakdown(
        await fetch_all(
            f"""
            SELECT COALESCE(tool_name, 'unknown') AS key, COUNT(*) AS count
            FROM tool_audit_logs tal
            WHERE datetime(tal.created_at) >= datetime('now', ?)
            {tool_kb_sql}
            GROUP BY COALESCE(tool_name, 'unknown')
            ORDER BY count DESC
            LIMIT 8
            """,
            (period, *tool_kb_params),
        )
    )
    job_status = _breakdown(
        await fetch_all(
            f"""
            SELECT COALESCE(status, 'unknown') AS key, COUNT(*) AS count
            FROM background_jobs bj
            WHERE datetime(bj.created_at) >= datetime('now', ?)
            {job_kb_sql}
            GROUP BY COALESCE(status, 'unknown')
            ORDER BY count DESC
            """,
            (period, *job_kb_params),
        )
    )
    support_status = _breakdown(
        await fetch_all(
            f"""
            SELECT COALESCE(workflow_status, status, 'unknown') AS key, COUNT(*) AS count
            FROM support_tickets st
            WHERE datetime(st.created_at) >= datetime('now', ?)
            {ticket_kb_sql}
            GROUP BY COALESCE(workflow_status, status, 'unknown')
            ORDER BY count DESC
            """,
            (period, *ticket_kb_params),
        )
    )
    support_intents = _breakdown(
        await fetch_all(
            f"""
            SELECT COALESCE(intent, issue_type, 'unknown') AS key, COUNT(*) AS count
            FROM support_tickets st
            WHERE datetime(st.created_at) >= datetime('now', ?)
            {ticket_kb_sql}
            GROUP BY COALESCE(intent, issue_type, 'unknown')
            ORDER BY count DESC
            LIMIT 8
            """,
            (period, *ticket_kb_params),
        )
    )
    pending_status = _breakdown(
        await fetch_all(
            f"""
            SELECT COALESCE(status, 'unknown') AS key, COUNT(*) AS count
            FROM pending_actions pa
            WHERE datetime(pa.created_at) >= datetime('now', ?)
            {pending_kb_sql}
            GROUP BY COALESCE(status, 'unknown')
            ORDER BY count DESC
            """,
            (period, *pending_kb_params),
        )
    )

    return AnalyticsDashboardOutput(
        period_days=int(days),
        kb_id=kb_id,
        summary=summary,
        timeseries=timeseries,
        chat_modes=chat_modes,
        kb_usage=_breakdown(kb_usage_rows),
        top_tools=top_tools,
        job_status=job_status,
        support_status=support_status,
        support_intents=support_intents,
        pending_status=pending_status,
    )
