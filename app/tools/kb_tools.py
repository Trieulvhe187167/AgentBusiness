"""
Knowledge-base-oriented tools.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

import app.rag as rag
from app.models import RequestContext
from app.tools.registry import ToolAuthPolicy, ToolSpec


class SearchKBInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    kb_id: int | None = Field(default=None, ge=1)
    kb_key: str | None = Field(default=None, min_length=1, max_length=80)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchKBHit(BaseModel):
    rank: int
    chunk_id: str
    filename: str
    file_type: str
    similarity: float
    category: str | None = None
    lang: str | None = None
    preview: str


class SearchKBOutput(BaseModel):
    query: str
    kb_id: int
    kb_key: str
    kb_name: str
    top_k: int
    total_hits: int
    hits: list[SearchKBHit]


async def _search_kb_tool(payload: SearchKBInput, context: RequestContext) -> dict[str, Any]:
    target_kb_id = payload.kb_id if payload.kb_id is not None else context.kb_id
    target_kb_key = payload.kb_key or context.kb_key
    kb_scope = rag._resolve_kb_scope(kb_id=target_kb_id, kb_key=target_kb_key)
    results = rag.retrieve(payload.query, top_k=payload.top_k, kb_id=kb_scope["id"])

    return {
        "query": payload.query,
        "kb_id": kb_scope["id"],
        "kb_key": kb_scope["key"],
        "kb_name": kb_scope["name"],
        "top_k": payload.top_k,
        "total_hits": len(results),
        "hits": [
            {
                "rank": index + 1,
                "chunk_id": str(item.get("chunk_id") or ""),
                "filename": item.get("filename") or "unknown",
                "file_type": item.get("file_type") or "unknown",
                "similarity": round(float(item.get("similarity", 0.0)), 6),
                "category": item.get("category"),
                "lang": item.get("lang"),
                "preview": (item.get("text") or "")[:280],
            }
            for index, item in enumerate(results)
        ],
    }


def build_search_kb_tool() -> ToolSpec:
    return ToolSpec(
        name="search_kb",
        description="Search within a scoped Knowledge Base and return ranked supporting chunks.",
        input_model=SearchKBInput,
        output_model=SearchKBOutput,
        auth_policy=ToolAuthPolicy(allow_anonymous=True, scope="kb"),
        timeout_seconds=15,
        idempotent=True,
        handler=_search_kb_tool,
        summarize_result=lambda payload: (
            f"search_kb returned {payload.get('total_hits', 0)} hit(s) for kb '{payload.get('kb_key', '')}'"
        ),
    )
