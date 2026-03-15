"""
Admin and diagnostic tools.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.database import fetch_one
from app.kb_service import KB_SELECT, open_db, resolve_kb_scope, row_to_kb_summary
from app.models import KBStats, KnowledgeBaseSummary, RequestContext
from app.tools.registry import ToolAuthPolicy, ToolSpec
from app.vector_store import vector_store


class EmptyToolInput(BaseModel):
    pass


class ListKBsOutput(BaseModel):
    total: int
    items: list[KnowledgeBaseSummary]


class GetKBStatsInput(BaseModel):
    kb_id: int | None = Field(default=None, ge=1)
    kb_key: str | None = Field(default=None, min_length=1, max_length=80)


async def _list_kbs_tool(_: EmptyToolInput, __: RequestContext) -> dict[str, Any]:
    db = await open_db()
    try:
        cursor = await db.execute(
            KB_SELECT
            + """
            GROUP BY
                kb.id, kb.key, kb.name, kb.description, kb.status,
                kb.is_default, kb.kb_version, kb.created_at, kb.updated_at
            ORDER BY kb.is_default DESC, kb.created_at ASC
            """
        )
        rows = await cursor.fetchall()
        items = [row_to_kb_summary(dict(row)).model_dump() for row in rows]
    finally:
        await db.close()

    return {"total": len(items), "items": items}


async def _get_kb_stats_tool(payload: GetKBStatsInput, context: RequestContext) -> dict[str, Any]:
    target_kb_id = payload.kb_id if payload.kb_id is not None else context.kb_id
    target_kb_key = payload.kb_key or context.kb_key
    db = await open_db()
    try:
        kb_scope = await resolve_kb_scope(db, kb_id=target_kb_id, kb_key=target_kb_key)
    finally:
        await db.close()

    row = await fetch_one(
        """
        SELECT
            COUNT(*) AS total_files,
            COALESCE(SUM(CASE WHEN status = 'ingested' THEN 1 ELSE 0 END), 0) AS ingested_files
        FROM kb_files
        WHERE kb_id = ?
        """,
        (kb_scope.id,),
    )
    where = {"kb_id": kb_scope.id}
    total_vectors = vector_store.count_by_where(where)
    return KBStats(
        total_files=int((row or {}).get("total_files") or 0),
        ingested_files=int((row or {}).get("ingested_files") or 0),
        total_chunks=total_vectors,
        total_vectors=total_vectors,
        sources=vector_store.get_sources(where),
        scope="kb",
        kb_id=kb_scope.id,
        kb_key=kb_scope.key,
        kb_name=kb_scope.name,
        kb_version=kb_scope.kb_version,
        is_default=kb_scope.is_default,
    ).model_dump()


def build_list_kbs_tool() -> ToolSpec:
    return ToolSpec(
        name="list_kbs",
        description="List all Knowledge Bases with counts and current status.",
        input_model=EmptyToolInput,
        output_model=ListKBsOutput,
        auth_policy=ToolAuthPolicy(required_roles=["admin"], scope="admin"),
        timeout_seconds=10,
        idempotent=True,
        handler=_list_kbs_tool,
        summarize_result=lambda payload: f"list_kbs returned {payload.get('total', 0)} KB(s)",
    )


def build_get_kb_stats_tool() -> ToolSpec:
    return ToolSpec(
        name="get_kb_stats",
        description="Return KB-level ingest and vector statistics for a selected Knowledge Base.",
        input_model=GetKBStatsInput,
        output_model=KBStats,
        auth_policy=ToolAuthPolicy(required_roles=["admin"], scope="admin"),
        timeout_seconds=10,
        idempotent=True,
        handler=_get_kb_stats_tool,
        summarize_result=lambda payload: (
            f"get_kb_stats returned {payload.get('total_vectors', 0)} vector(s) for kb '{payload.get('kb_key', '')}'"
        ),
    )
