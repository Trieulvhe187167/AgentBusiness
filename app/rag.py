"""
RAG orchestration with 3-mode logic, multilingual support,
query expansion, BM25 reranking, numeric guardrail, and conversational memory.
"""

from __future__ import annotations

import contextvars
import json
import hashlib
import logging
import re
import time
import uuid
from collections.abc import Mapping
from typing import Any, Generator

from app.cache import (
    get_cached_embedding,
    get_cached_response_payload,
    get_cached_retrieval,
    get_semantic_cached_response,
    get_semantic_cached_retrieval,
    set_cached_embedding,
    set_cached_response_payload,
    set_cached_retrieval,
    set_semantic_cached_response,
    set_semantic_cached_retrieval,
)
from app.config import settings
from app.conversation_memory import build_conversation_context, load_recent_turns, resolve_followup_query
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.embeddings import embed_query, using_hashing_fallback
from app.kb_service import ensure_kb_access, normalize_kb_key
from app.knowledge_gaps import record_knowledge_gap
from app.lang import detect_language
from app.llm_client import (
    active_provider_name,
    complete_chat,
    generate_stream,
    get_last_generation_usage,
    is_llm_ready,
    reset_last_generation_usage,
)
from app.models import Citation, RequestContext
from app.observability import retrieval_trace_attrs, trace_span
from app.query_expander import expand_query
from app.reranker import rerank, rerank_candidate_limit
from app.runtime_controls import (
    LatencyTracker,
    budget_snapshot,
    corrective_disabled_for_request,
    effective_llm_latency_budget_ms,
    effective_max_answer_chunks,
    effective_max_rerank_candidates,
    effective_retrieval_latency_budget_ms,
    reranker_disabled_for_request,
)
from app.vector_store import vector_store

logger = logging.getLogger(__name__)
_latency_tracker_var: contextvars.ContextVar[LatencyTracker | None] = contextvars.ContextVar(
    "rag_latency_tracker",
    default=None,
)

SYSTEM_PROMPT = """
You are a customer support assistant.
Rules:
1) Answer strictly from CONTEXT. Do NOT invent facts, numbers, or dates not present in the context.
2) If information is missing, say you cannot find it in the provided documents.
3) Keep answers concise and factual (1-3 sentences for direct questions).
4) Respond in the user's language.
""".strip()

CONTEXT_TEMPLATE = """--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}
"""

DEFAULT_CLARIFY = {
    "vi": "Bạn có thể mô tả rõ hơn về điều bạn cần biết không? Mình sẽ tìm kiếm chính xác hơn.",
    "en": "Could you describe what you're looking for in a bit more detail? That will help me find the right information.",
}

# Keys are matched against the retrieved chunk's `category` field (case-insensitive substring).
# Add more domain-specific entries here as needed — these work for any data type.
CLARIFY_BY_CATEGORY: dict[str, dict[str, str]] = {
    # --- logistics / e-commerce ---
    "shipping": {
        "vi": "Bạn ở khu vực nào hoặc cần thêm thông tin gì về vận chuyển?",
        "en": "Which area or what specific delivery detail are you looking for?",
    },
    "giao": {
        "vi": "Bạn ở khu vực nào hoặc cần thêm thông tin gì về vận chuyển?",
        "en": "Which area or what specific delivery detail are you looking for?",
    },
    "return": {
        "vi": "Bạn cần đổi, trả hay hoàn tiền? Và thông tin nào bạn cần biết thêm?",
        "en": "Are you looking for exchange, return, or refund information? Any specific detail?",
    },
    "đổi": {
        "vi": "Bạn cần đổi, trả hay hoàn tiền? Và thông tin nào bạn cần biết thêm?",
        "en": "Are you looking for exchange, return, or refund information? Any specific detail?",
    },
    "payment": {
        "vi": "Bạn muốn biết thêm về hình thức hoặc quy trình thanh toán nào?",
        "en": "Which payment method or process would you like more details on?",
    },
    "thanh toán": {
        "vi": "Bạn muốn biết thêm về hình thức hoặc quy trình thanh toán nào?",
        "en": "Which payment method or process would you like more details on?",
    },
    # --- education / university ---
    "admission": {
        "vi": "Bạn đang hỏi về tuyển sinh năm nào hoặc ngành học nào?",
        "en": "Which year or program are you asking about for admissions?",
    },
    "tuyển sinh": {
        "vi": "Bạn đang hỏi về tuyển sinh năm nào hoặc ngành học nào?",
        "en": "Which year or program are you asking about for admissions?",
    },
    "course": {
        "vi": "Bạn cần thông tin về môn học cụ thể nào hoặc chương trình nào?",
        "en": "Which course or program are you looking for information about?",
    },
    "môn học": {
        "vi": "Bạn cần thông tin về môn học cụ thể nào hoặc chương trình nào?",
        "en": "Which course or program are you looking for information about?",
    },
    # --- HR / employee ---
    "employee": {
        "vi": "Bạn muốn biết thêm về nhân viên nào hoặc phòng ban nào?",
        "en": "Which employee or department are you asking about?",
    },
    "nhân viên": {
        "vi": "Bạn muốn biết thêm về nhân viên nào hoặc phòng ban nào?",
        "en": "Which employee or department are you asking about?",
    },
    # --- customer / CRM ---
    "customer": {
        "vi": "Bạn cần tìm thông tin khách hàng theo tiêu chí nào (tên, mã, khu vực…)?",
        "en": "What detail are you looking up for the customer (name, ID, region…)?",
    },
    "khách hàng": {
        "vi": "Bạn cần tìm thông tin khách hàng theo tiêu chí nào (tên, mã, khu vực…)?",
        "en": "What detail are you looking up for the customer (name, ID, region…)?",
    },
    # --- product / catalog ---
    "product": {
        "vi": "Bạn đang tìm hiểu về sản phẩm hay dòng hàng cụ thể nào?",
        "en": "Which product or category would you like to know more about?",
    },
    "sản phẩm": {
        "vi": "Bạn đang tìm hiểu về sản phẩm hay dòng hàng cụ thể nào?",
        "en": "Which product or category would you like to know more about?",
    },
    # --- policy / regulation ---
    "policy": {
        "vi": "Bạn muốn biết về chính sách nào cụ thể?",
        "en": "Which specific policy are you asking about?",
    },
    "quy định": {
        "vi": "Bạn muốn biết về quy định nào cụ thể?",
        "en": "Which specific regulation or rule are you asking about?",
    },
}

FALLBACK_TEXT = {
    "vi": (
        "Mình chưa tìm thấy thông tin phù hợp trong tài liệu hiện có. "
        "Nếu bạn có thể cung cấp thêm từ khóa hoặc ngữ cảnh, mình sẽ thử tìm lại."
        "{topics_hint}"
    ),
    "en": (
        "I couldn't find relevant information in the current documents. "
        "Try rephrasing your question or providing more context."
        "{topics_hint}"
    ),
}

CORRECTIVE_REWRITE_SYSTEM_PROMPT = """
You rewrite weak retrieval queries for an internal business knowledge base.
Return one compact JSON object only. Do not answer the user.
The JSON object must be {"query":"..."}.
Keep the query in the user's language unless translation is necessary for retrieval.
Preserve product codes, order IDs, dates, numbers, names, and policy terms exactly.
""".strip()

CORRECTIVE_REWRITE_TEMPLATE = """The first retrieval attempt was weak.

Reason: {reason}
User language: {lang}
Original user question:
{query}

Top retrieved snippets, if any:
{snippets}

Rewrite the search query to retrieve better internal KB passages."""

# Regex to detect numbers in text (for guardrail)
_NUMBER_RE = re.compile(r"\b\d[\d,\.\s]*\d|\b\d\b")


def _resolve_kb_scope(
    kb_id: int | None = None,
    kb_key: str | None = None,
    *,
    auth_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if kb_id is not None:
        row = fetch_one_sync(
            """
            SELECT id, key, name, kb_version, is_default, access_level, tenant_id, org_id
            FROM knowledge_bases
            WHERE id = ?
            """,
            (kb_id,),
        )
    elif kb_key:
        row = fetch_one_sync(
            """
            SELECT id, key, name, kb_version, is_default, access_level, tenant_id, org_id
            FROM knowledge_bases
            WHERE key = ?
            """,
            (normalize_kb_key(kb_key),),
        )
    else:
        row = fetch_one_sync(
            """
            SELECT id, key, name, kb_version, is_default, access_level, tenant_id, org_id
            FROM knowledge_bases
            WHERE is_default = 1
            LIMIT 1
            """
        )

    if not row:
        target = f"id={kb_id}" if kb_id is not None else f"key={kb_key}" if kb_key else "default"
        raise ValueError(f"Knowledge Base not found for {target}")
    if auth_context is not None:
        ensure_kb_access(row, auth_context)

    return {
        "id": int(row["id"]),
        "key": row["key"],
        "name": row["name"],
        "kb_version": row["kb_version"],
        "is_default": bool(row.get("is_default")),
        "access_level": str(row.get("access_level") or "public"),
        "tenant_id": row.get("tenant_id"),
        "org_id": row.get("org_id"),
    }


def _scoped_session_id(session_id: str, kb_id: int) -> str:
    return f"{session_id}::kb:{kb_id}"


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def _embedding_for_query(query: str) -> list[float]:
    normalized = _normalize_query(query)
    cached_emb = get_cached_embedding(normalized)
    query_embedding = cached_emb if cached_emb is not None else embed_query(normalized)
    if cached_emb is None:
        set_cached_embedding(normalized, query_embedding)
    return query_embedding


def _semantic_cache_available() -> bool:
    return not using_hashing_fallback()


def _build_auth_cache_scope(auth_context: dict[str, Any] | None) -> str:
    auth = auth_context or {}
    roles = sorted(str(role).strip().lower() for role in (auth.get("roles") or []) if str(role).strip())
    parts = [
        f"channel:{auth.get('channel') or 'web'}",
        f"roles:{','.join(roles)}",
        f"tenant:{auth.get('tenant_id') or ''}",
        f"org:{auth.get('org_id') or ''}",
    ]
    return "|".join(parts)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for item in results:
        key = "|".join([
            str(item.get("filename", "")),
            str(item.get("row_num", "")),
            str(item.get("page_num", "")),
            str(item.get("sheet_name", "")),
        ])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _diversification_key(item: dict[str, Any]) -> str:
    return str(item.get("source_id") or item.get("file_id") or item.get("filename") or item.get("chunk_id") or "")


def _source_diversified(results: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], bool]:
    if (
        not settings.retrieval_source_diversification_enabled
        or settings.effective_retrieval_source_max_chunks_per_source <= 0
        or limit <= 0
    ):
        return results[:limit], False

    max_per_source = settings.effective_retrieval_source_max_chunks_per_source
    counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    diversified = False
    for item in results:
        key = _diversification_key(item)
        current = counts.get(key, 0)
        if key and current >= max_per_source:
            deferred.append(item)
            diversified = True
            continue
        selected.append(item)
        if key:
            counts[key] = current + 1
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for item in deferred:
            selected.append(item)
            if len(selected) >= limit:
                break

    if diversified:
        for item in selected:
            item["source_diversified"] = True
    return selected[:limit], diversified


def _source_label(item: dict[str, Any]) -> str:
    source = item.get("filename", "document")
    if item.get("page_num"):
        source += f" p.{item['page_num']}"
    if item.get("sheet_name"):
        source += f"/{item['sheet_name']}"
    if item.get("row_num"):
        source += f" row {item['row_num']}"
    return source


def _build_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for item in results[: settings.max_citations]:
        row_ref = str(item.get("row_num")) if item.get("row_num") is not None else None
        citation = Citation(
            filename=item.get("filename", "unknown"),
            file_type=item.get("file_type", "unknown"),
            page_num=item.get("page_num"),
            sheet_name=item.get("sheet_name"),
            row_range=row_ref,
            content_preview=item.get("content_preview") or item.get("text", "")[:220],
            chunk_id=item.get("chunk_id", ""),
            score=round(float(item.get("similarity", 0.0)), 4),
        )
        cards.append(citation.model_dump())
    return cards


def _extract_clean_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    marker = "keywords:"
    lower = text.lower()
    marker_idx = lower.find(marker)
    if marker_idx != -1:
        text = text[:marker_idx].strip()
    colon_idx = text.find(": ")
    if 0 < colon_idx < 90:
        text = text[colon_idx + 2:].strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extractive_answer(results: list[dict[str, Any]], lang: str) -> str:
    top = results[: max(settings.max_extractive_chunks, 1)]
    if not top:
        return "I could not find relevant information." if lang == "en" else "Mình không tìm thấy thông tin phù hợp."

    best = _extract_clean_text(top[0].get("text", ""))
    extra_lines = []
    for item in top[1:]:
        extra = _extract_clean_text(item.get("text", ""))
        if extra and extra != best:
            extra_lines.append(f"- {extra}")

    parts = [best] if best else []
    parts.extend(extra_lines)

    if settings.debug_show_retrieval:
        parts.append("\n[debug retrieval]")
        for idx, item in enumerate(results, start=1):
            parts.append(f"{idx}. {_source_label(item)} score={item.get('similarity', 0.0):.4f}")

    return "\n".join(parts).strip()


def _llm_failure_note(lang: str) -> str:
    if lang == "vi":
        return "\n\n[He thong da chuyen sang tra loi tu trich dan do LLM tam thoi cham/khong san sang.]"
    return "\n\n[Switched to citation-based answer because the LLM is temporarily slow or unavailable.]"


def _stream_llm_with_extract_fallback(
    *,
    full_prompt: str,
    results: list[dict[str, Any]],
    lang: str,
) -> tuple[str, bool, dict[str, int]]:
    generated_parts: list[str] = []
    reset_last_generation_usage()
    tracker = _latency_tracker_var.get()
    llm_started = time.perf_counter()
    try:
        for token in generate_stream(full_prompt, system_prompt=SYSTEM_PROMPT):
            generated_parts.append(token)
            yield {"event": "token", "data": {"text": token}}
    except Exception as err:
        if tracker is not None:
            tracker.add("llm_ms", llm_started)
        logger.warning("LLM generation failed, falling back to extractive answer: %s", err)
        extractive = _extractive_answer(results, lang)
        if generated_parts:
            fallback_text = _llm_failure_note(lang) + "\n" + extractive
            yield {"event": "token", "data": {"text": fallback_text}}
            return "".join(generated_parts).strip() + fallback_text, False, get_last_generation_usage()
        yield {"event": "token", "data": {"text": extractive}}
        return extractive, False, get_last_generation_usage()

    if tracker is not None:
        tracker.add("llm_ms", llm_started)
        llm_budget_ms = effective_llm_latency_budget_ms()
        if llm_budget_ms and tracker.llm_ms > llm_budget_ms:
            tracker.event("llm_budget_exceeded", budget_ms=llm_budget_ms, actual_ms=tracker.llm_ms)

    answer_text = "".join(generated_parts).strip()
    llm_usage = get_last_generation_usage()
    if _answer_has_hallucinated_numbers(answer_text, _context_for_llm(results)):
        logger.warning("Guardrail triggered: falling back to extractive answer")
        corrected = _extractive_answer(results, lang)
        yield {"event": "token", "data": {"text": "\n[corrected]\n" + corrected}}
        return corrected, False, llm_usage
    return answer_text, True, llm_usage


def _fallback_text(results: list[dict[str, Any]], lang: str) -> str:
    """If partial results exist, surface the best chunk. Otherwise explain what's available."""
    # Try to give partial answer from available chunks
    if results:
        best = _extract_clean_text(results[0].get("text", ""))
        if best:
            note = (
                "\n\n_(Lưu ý: Thông tin tìm được có thể chưa đầy đủ – bạn có thể cung cấp thêm chi tiết để mình hỗ trợ tốt hơn.)_"
                if lang == "vi"
                else "\n\n_(Note: This is the closest information found – you may provide more detail for a better answer.)_"
            )
            return best + note

    categories, seen = [], set()
    for item in results:
        cat = (item.get("category") or "").strip()
        if not cat:
            continue
        key = cat.lower()
        if key in seen:
            continue
        seen.add(key)
        categories.append(cat)

    # Build topics hint dynamically from actual data categories (no hardcoded domain)
    topics_hint = ""
    if categories:
        topics_hint = (
            f"\n\nDữ liệu hiện có chứa thông tin về: {', '.join(categories[:6])}."
            if lang == "vi"
            else f"\n\nThe available data contains information about: {', '.join(categories[:6])}."
        )
    return FALLBACK_TEXT[lang].format(topics_hint=topics_hint)


def _clarify_question(results: list[dict[str, Any]], lang: str) -> str:
    for item in results:
        category = (item.get("category") or "").lower()
        for key, template in CLARIFY_BY_CATEGORY.items():
            if key in category:
                return template[lang]
    return DEFAULT_CLARIFY[lang]


def _context_for_llm(results: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, item in enumerate(results[: effective_max_answer_chunks()], start=1):
        blocks.append(f"[{idx}] Source: {_source_label(item)}\n{item.get('text', '')}")
    return "\n\n".join(blocks)


def _effective_corrective_threshold() -> float:
    threshold = settings.effective_corrective_rag_min_score
    if vector_store.backend_name == "qdrant" and settings.qdrant_hybrid_enabled and settings.corrective_rag_min_score <= 0:
        threshold = settings.qdrant_hybrid_threshold_low
    if using_hashing_fallback() and settings.corrective_rag_min_score <= 0:
        threshold = min(threshold, settings.hashing_threshold_low)
    return float(threshold)


def _corrective_trigger_reason(results: list[dict[str, Any]], top_score: float) -> str | None:
    if not settings.corrective_rag_enabled or settings.effective_corrective_rag_max_attempts <= 0:
        return None
    if not results:
        return "no_results"
    if len(results) < settings.effective_corrective_rag_min_results:
        return "too_few_results"
    if top_score < _effective_corrective_threshold():
        return "low_top_score"
    return None


def _rewrite_snippet_preview(results: list[dict[str, Any]], limit: int = 3) -> str:
    if not results:
        return "-"
    lines = []
    for idx, item in enumerate(results[:limit], start=1):
        source = _source_label(item)
        text = re.sub(r"\s+", " ", str(item.get("text") or item.get("content_preview") or "")).strip()
        lines.append(f"{idx}. {source}: {text[:500]}")
    return "\n".join(lines)


def _extract_rewritten_query(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return text[:300].strip().strip('"')
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("query") or "").strip()


def _rewrite_query_for_correction(
    query: str,
    *,
    results: list[dict[str, Any]],
    lang: str,
    reason: str,
) -> tuple[str | None, str | None]:
    if not is_llm_ready():
        return None, "llm_unavailable"
    prompt = CORRECTIVE_REWRITE_TEMPLATE.format(
        reason=reason,
        lang=lang,
        query=query,
        snippets=_rewrite_snippet_preview(results),
    )
    try:
        response = complete_chat(
            prompt,
            system_prompt=CORRECTIVE_REWRITE_SYSTEM_PROMPT,
            timeout_seconds=settings.effective_corrective_rag_rewrite_timeout_seconds,
            max_tokens=settings.effective_corrective_rag_rewrite_max_tokens,
            response_format={"type": "json_object"},
        )
        rewritten = _normalize_query(_extract_rewritten_query(response.text))
    except Exception as err:
        logger.warning("Corrective RAG query rewrite failed: %s", err)
        return None, f"rewrite_error:{err.__class__.__name__}"

    if not rewritten:
        return None, "empty_rewrite"
    if rewritten.lower() == _normalize_query(query).lower():
        return None, "unchanged_rewrite"
    return rewritten, None


def _maybe_corrective_retrieve(
    *,
    query: str,
    results: list[dict[str, Any]],
    top_score: float,
    lang: str,
    kb_id: int,
    auth_context: dict[str, Any] | None,
    runtime_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "enabled": bool(settings.corrective_rag_enabled) and not corrective_disabled_for_request(runtime_context),
        "attempt_count": 1,
        "query_rewritten": False,
        "correction_reason": None,
        "rewrite_error": None,
        "rewritten_query": None,
    }
    if corrective_disabled_for_request(runtime_context):
        return results, metadata
    reason = _corrective_trigger_reason(results, top_score)
    if reason is None:
        return results, metadata

    metadata["correction_reason"] = reason
    rewritten_query, rewrite_error = _rewrite_query_for_correction(
        query,
        results=results,
        lang=lang,
        reason=reason,
    )
    if rewrite_error:
        metadata["rewrite_error"] = rewrite_error
        return results, metadata
    if not rewritten_query:
        return results, metadata

    retried = retrieve(
        rewritten_query,
        kb_id=kb_id,
        auth_context=auth_context,
        runtime_context=runtime_context,
    )
    retried = _apply_lang_boost(retried, lang)
    retried_top_score = float(retried[0].get("similarity", 0.0)) if retried else 0.0
    if retried and (not results or retried_top_score >= top_score):
        metadata.update(
            {
                "attempt_count": 2,
                "query_rewritten": True,
                "rewritten_query": rewritten_query,
                "previous_top_score": round(float(top_score), 4),
                "corrected_top_score": round(float(retried_top_score), 4),
            }
        )
        return retried, metadata

    metadata.update(
        {
            "attempt_count": 2,
            "query_rewritten": False,
            "rewritten_query": rewritten_query,
            "rewrite_error": "retried_results_not_better",
            "previous_top_score": round(float(top_score), 4),
            "corrected_top_score": round(float(retried_top_score), 4),
        }
    )
    return results, metadata


# ── Conversational memory (rule-based) ───────────────────────────────────────

def _load_recent_turns(session_id: str, n: int = 3) -> list[dict[str, Any]]:
    """Fetch the last n chat turns for the session."""
    return load_recent_turns(session_id, limit=n)


def _build_memory_summary(turns: list[dict[str, Any]]) -> str:
    return build_conversation_context(turns)


# ── Numeric guardrail ─────────────────────────────────────────────────────────

def _normalize_number_str(s: str) -> str:
    """Strip thousands separators to get bare numeric string."""
    return re.sub(r"[,\.\s]", "", s)


def _answer_has_hallucinated_numbers(answer: str, context: str) -> bool:
    """
    Returns True if the answer contains numeric values not present in the context.
    """
    answer_nums = _NUMBER_RE.findall(answer)
    if not answer_nums:
        return False  # no numbers → nothing to check

    context_normalized = _normalize_number_str(context)
    for num_str in answer_nums:
        bare = _normalize_number_str(num_str)
        if len(bare) < 2:
            continue  # skip single digits
        if bare not in context_normalized:
            logger.warning("Guardrail: hallucinated number '%s' not in context", bare)
            return True
    return False


# ── 3-mode threshold ──────────────────────────────────────────────────────────

def decide_mode(top_score: float) -> str:
    """
    Decide answer mode with adjusted thresholds.
    'clarify' is treated as a soft 'answer' — we still try to answer before asking.
    Only true fallback (no relevant results) triggers the 'fallback' path.
    """
    threshold_good = settings.threshold_good
    threshold_low = settings.threshold_low
    if vector_store.backend_name == "qdrant" and settings.qdrant_hybrid_enabled:
        threshold_good = settings.qdrant_hybrid_threshold_good
        threshold_low = settings.qdrant_hybrid_threshold_low
    if using_hashing_fallback():
        threshold_good = min(threshold_good, settings.hashing_threshold_good)
        threshold_low = min(threshold_low, settings.hashing_threshold_low)

    if top_score >= threshold_good:
        return "answer"
    if top_score >= threshold_low:
        # Return 'clarify' only when LLM is unavailable; otherwise try to answer.
        return "clarify"
    return "fallback"


# ── Session helpers ───────────────────────────────────────────────────────────

def _load_pending_session(session_id: str) -> dict[str, Any] | None:
    return fetch_one_sync("SELECT * FROM chat_sessions WHERE session_id=?", (session_id,))


def _save_pending_session(session_id: str, query: str, category: str, lang: str):
    execute_sync(
        """
        INSERT INTO chat_sessions (
            session_id, pending_clarify_query,
            pending_clarify_category, pending_clarify_lang, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            pending_clarify_query=excluded.pending_clarify_query,
            pending_clarify_category=excluded.pending_clarify_category,
            pending_clarify_lang=excluded.pending_clarify_lang,
            updated_at=excluded.updated_at
        """,
        (session_id, query, category, lang, utcnow_iso()),
    )


def _clear_pending_session(session_id: str):
    execute_sync(
        """
        UPDATE chat_sessions
        SET pending_clarify_query=NULL,
            pending_clarify_category=NULL,
            pending_clarify_lang=NULL,
            updated_at=?
        WHERE session_id=?
        """,
        (utcnow_iso(), session_id),
    )


def _coerce_request_context(request_context: RequestContext | dict[str, Any] | None) -> dict[str, Any]:
    if request_context is None:
        return {}
    if isinstance(request_context, RequestContext):
        payload = request_context.model_dump()
    elif hasattr(request_context, "model_dump"):
        payload = request_context.model_dump()
    elif isinstance(request_context, Mapping):
        payload = dict(request_context)
    else:
        logger.warning("Ignoring unexpected request_context type: %s", type(request_context).__name__)
        return {}

    auth = dict(payload.get("auth") or {})
    auth["roles"] = list(auth.get("roles") or [])
    auth["channel"] = auth.get("channel") or "web"
    payload["auth"] = auth
    return payload


def _log_chat(
    session_id: str,
    user_message: str,
    merged_query: str,
    mode: str,
    top_score: float,
    answer_text: str,
    citations: list[dict[str, Any]],
    latency_ms: int,
    llm_provider: str,
    request_context: RequestContext | dict[str, Any] | None = None,
    llm_usage: dict[str, int] | None = None,
) -> int | None:
    context = _coerce_request_context(request_context)
    auth = context.get("auth") or {}
    usage = llm_usage or {}
    return execute_sync(
        """
        INSERT INTO chat_logs (
            session_id, request_id, user_id, roles_json, channel, tenant_id, org_id,
            kb_id, kb_key, user_message, merged_query, mode, top_score,
            answer_text, citations_json, latency_ms, llm_provider,
            llm_input_tokens, llm_output_tokens, llm_total_tokens, llm_cached_tokens,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            context.get("request_id"),
            auth.get("user_id"),
            json.dumps(auth.get("roles") or [], ensure_ascii=False),
            auth.get("channel"),
            auth.get("tenant_id"),
            auth.get("org_id"),
            context.get("kb_id"),
            context.get("kb_key"),
            user_message,
            merged_query,
            mode,
            float(top_score), answer_text,
            json.dumps(citations, ensure_ascii=False),
            latency_ms,
            llm_provider,
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
            int(usage.get("total_tokens") or 0),
            int(usage.get("cached_tokens") or 0),
            utcnow_iso(),
        ),
    )


# ── Core retrieve ─────────────────────────────────────────────────────────────

def _build_retrieval_scope(
    kb_scope: dict[str, Any],
    top_k: int,
    where: dict[str, Any] | None = None,
    auth_scope: str = "",
) -> str:
    where_parts: list[str] = []
    if where:
        where_parts = [f"{key}={where[key]}" for key in sorted(where)]
    return ":".join(
        [
            f"kb:{kb_scope['id']}",
            f"v:{kb_scope['kb_version']}",
            f"k:{top_k}",
            f"backend:{vector_store.backend_name}",
            f"retrieval_mode:{vector_store.retrieval_mode}",
            f"embedding:{settings.effective_embedding_fingerprint}",
            f"auth:{auth_scope}",
            f"where:{'|'.join(where_parts)}",
        ]
    )


def _build_response_cache_scope(
    *,
    kb_scope: dict[str, Any],
    lang: str,
    context: dict[str, Any],
    llm_provider: str,
) -> str:
    return ":".join(
        [
            "response:v1",
            f"kb:{kb_scope['id']}",
            f"v:{kb_scope['kb_version']}",
            f"access:{kb_scope['access_level']}",
            f"tenant:{kb_scope.get('tenant_id') or ''}",
            f"org:{kb_scope.get('org_id') or ''}",
            f"auth:{_build_auth_cache_scope(context.get('auth') or {})}",
            f"lang:{lang}",
            f"answer_mode:{settings.normalized_answer_mode}",
            f"provider:{llm_provider}",
            f"model:{settings.effective_chat_model or ''}",
            f"embedding:{settings.effective_embedding_fingerprint}",
            f"backend:{vector_store.backend_name}",
            f"retrieval_mode:{vector_store.retrieval_mode}",
            f"top_k:{settings.top_k}",
            f"max_answer_chunks:{effective_max_answer_chunks()}",
            f"corrective:{bool(settings.corrective_rag_enabled)}",
            f"corrective_min_score:{settings.effective_corrective_rag_min_score}",
            f"corrective_min_results:{settings.effective_corrective_rag_min_results}",
            f"system:{_hash_text(SYSTEM_PROMPT)}",
        ]
    )


def _response_cache_allowed(*, followup: bool, recent_turns: list[dict[str, Any]]) -> bool:
    if not settings.response_cache_enabled:
        return False
    if followup or recent_turns:
        return False
    if settings.debug_show_retrieval:
        return False
    return True


def _get_response_cache_hit(query: str, scope: str) -> dict[str, Any] | None:
    payload = get_cached_response_payload(query, scope)
    if payload is not None:
        return {"type": "exact", "payload": payload}

    if not settings.semantic_response_cache_enabled or not _semantic_cache_available():
        return None
    query_embedding = _embedding_for_query(query)
    hit = get_semantic_cached_response(query_embedding, scope)
    if hit is None:
        return None
    return {
        "type": "semantic",
        "payload": hit["payload"],
        "score": hit.get("score"),
        "query": hit.get("query"),
    }


def _set_response_cache(query: str, scope: str, payload: dict[str, Any]):
    set_cached_response_payload(query, scope, payload)
    if settings.semantic_response_cache_enabled and _semantic_cache_available():
        query_embedding = _embedding_for_query(query)
        set_semantic_cached_response(query, query_embedding, scope, payload)


def _retrieve_single(query: str, top_k: int, where: dict[str, Any], cache_scope: str) -> list[dict[str, Any]]:
    """Retrieve for a single query string (with embedding cache)."""
    normalized = _normalize_query(query)
    tracker = _latency_tracker_var.get()
    embed_started = time.perf_counter()
    query_embedding = _embedding_for_query(normalized)
    if tracker is not None:
        tracker.add("embedding_ms", embed_started)

    cached_results = get_cached_retrieval(normalized, cache_scope)
    if cached_results is not None:
        logger.debug("Retrieval cache hit: type=exact scope=%s query=%s", cache_scope, normalized)
        if tracker is not None:
            tracker.cache_hit = True
            tracker.event("retrieval_cache_hit", cache_type="exact")
        return cached_results

    if settings.semantic_retrieval_cache_enabled and _semantic_cache_available():
        semantic_hit = get_semantic_cached_retrieval(query_embedding, cache_scope)
        if semantic_hit is not None:
            logger.debug(
                "Retrieval cache hit: type=semantic score=%.4f cached_query=%s",
                float(semantic_hit.get("score") or 0.0),
                semantic_hit.get("query"),
            )
            if tracker is not None:
                tracker.cache_hit = True
                tracker.event("retrieval_cache_hit", cache_type="semantic", score=semantic_hit.get("score"))
            return semantic_hit["results"]

    vector_started = time.perf_counter()
    raw = vector_store.query(query_embedding, top_k=top_k, where=where, query_text=normalized)
    if tracker is not None:
        tracker.add("vector_query_ms", vector_started)
    results = raw or []
    set_cached_retrieval(normalized, cache_scope, results)
    if settings.semantic_retrieval_cache_enabled and _semantic_cache_available():
        set_semantic_cached_retrieval(normalized, query_embedding, cache_scope, results)
    return results


def retrieve(
    query: str,
    top_k: int | None = None,
    *,
    kb_id: int | None = None,
    kb_key: str | None = None,
    auth_context: dict[str, Any] | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Multi-query retrieval: query original + expanded variants.
    Results are merged, deduplicated, filtered by min_similarity, and reranked.
    """
    k = top_k or settings.top_k
    reranker_disabled = reranker_disabled_for_request(runtime_context)
    candidate_k = (
        int(k)
        if reranker_disabled
        else max(int(k), min(rerank_candidate_limit(k), effective_max_rerank_candidates()))
    )
    tracker = _latency_tracker_var.get() or LatencyTracker()
    token = _latency_tracker_var.set(tracker)
    try:
        with trace_span(
            "rag.retrieve",
            retrieval_trace_attrs(query=query, top_k=candidate_k, kb_id=kb_id, kb_key=kb_key),
        ) as span:
            kb_scope = _resolve_kb_scope(kb_id=kb_id, kb_key=kb_key, auth_context=auth_context)
            span.set_attribute("rag.kb_id", kb_scope["id"])
            span.set_attribute("rag.kb_key", kb_scope["key"])
            where = {
                "kb_id": kb_scope["id"],
                "access_level": kb_scope["access_level"],
            }
            if kb_scope.get("tenant_id"):
                where["tenant_id"] = kb_scope["tenant_id"]
            if kb_scope.get("org_id"):
                where["org_id"] = kb_scope["org_id"]
            cache_scope = _build_retrieval_scope(
                kb_scope,
                candidate_k,
                where=where,
                auth_scope=_build_auth_cache_scope(auth_context),
            )
            variants = expand_query(query)
            span.set_attribute("rag.query_variant_count", len(variants))
            span.set_attribute("rag.cache.semantic_retrieval_enabled", bool(settings.semantic_retrieval_cache_enabled))

            all_raw: list[dict[str, Any]] = []
            for variant in variants:
                all_raw.extend(_retrieve_single(variant, candidate_k, where=where, cache_scope=cache_scope))

            floor = settings.min_similarity_threshold
            if vector_store.backend_name == "qdrant" and settings.qdrant_hybrid_enabled:
                floor = settings.qdrant_hybrid_min_similarity_threshold
            if using_hashing_fallback():
                floor = min(floor, settings.hashing_min_similarity_threshold)

            filtered = [item for item in all_raw if float(item.get("similarity", 0.0)) >= floor]
            deduped = _deduplicate(filtered)

            if reranker_disabled:
                reranked = sorted(
                    [dict(item) for item in deduped],
                    key=lambda item: float(item.get("similarity", 0.0)),
                    reverse=True,
                )
                for item in reranked:
                    item["reranker_provider"] = "disabled"
                    item["retrieval_score"] = round(float(item.get("similarity", 0.0)), 6)
                    item["final_score"] = item["retrieval_score"]
            else:
                rerank_started = time.perf_counter()
                reranked = rerank(query, deduped)
                tracker.add("reranker_ms", rerank_started)

            results, diversified = _source_diversified(reranked, k)
            retrieval_total_ms = tracker.embedding_ms + tracker.vector_query_ms + tracker.reranker_ms
            retrieval_budget_ms = effective_retrieval_latency_budget_ms()
            if retrieval_budget_ms and retrieval_total_ms > retrieval_budget_ms:
                tracker.event(
                    "retrieval_budget_exceeded",
                    budget_ms=retrieval_budget_ms,
                    actual_ms=retrieval_total_ms,
                )

            span.set_attribute("rag.final_top_k", k)
            span.set_attribute("rag.candidate_top_k", candidate_k)
            span.set_attribute("rag.runtime.max_rerank_candidates", effective_max_rerank_candidates())
            span.set_attribute("rag.runtime.reranker_disabled", bool(reranker_disabled))
            span.set_attribute("rag.reranker_provider", "disabled" if reranker_disabled else settings.normalized_reranker_provider)
            span.set_attribute("rag.source_diversification_enabled", bool(settings.retrieval_source_diversification_enabled))
            span.set_attribute("rag.source_diversification_applied", bool(diversified))
            span.set_attribute(
                "rag.source_max_chunks_per_source",
                settings.effective_retrieval_source_max_chunks_per_source,
            )
            span.set_attribute("rag.latency.embedding_ms", tracker.embedding_ms)
            span.set_attribute("rag.latency.vector_query_ms", tracker.vector_query_ms)
            span.set_attribute("rag.latency.reranker_ms", tracker.reranker_ms)
            span.set_attribute("rag.raw_result_count", len(all_raw))
            span.set_attribute("rag.filtered_result_count", len(filtered))
            span.set_attribute("rag.result_count", len(results))
            span.set_attribute("rag.top_score", float(results[0].get("similarity", 0.0)) if results else 0.0)
            return results
    finally:
        _latency_tracker_var.reset(token)


def _apply_lang_boost(results: list[dict[str, Any]], user_lang: str) -> list[dict[str, Any]]:
    """
    Proportional boost (+5%) for chunks in the same language as the user.
    Falls back to cross-lingual if too few same-lang chunks.
    """
    same_lang = sum(1 for r in results if r.get("lang") == user_lang)
    if same_lang < 2:
        # Not enough same-lang content → skip boost to allow cross-lingual
        return results

    for item in results:
        if item.get("lang") == user_lang:
            item["similarity"] = round(float(item.get("similarity", 0.0)) * 1.05, 6)

    results.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
    return results


# ── Main RAG stream ───────────────────────────────────────────────────────────

def rag_stream(
    query: str,
    session_id: str | None = None,
    lang: str | None = None,
    kb_id: int | None = None,
    kb_key: str | None = None,
    request_context: RequestContext | dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    start_time = time.perf_counter()
    user_query = _normalize_query(query)
    resolved_lang = detect_language(user_query, explicit_lang=lang)
    sid = session_id or uuid.uuid4().hex

    merged_query = user_query
    prompt_query = user_query
    followup = False
    scoped_sid = sid
    context = _coerce_request_context(request_context)
    latency_tracker = LatencyTracker()
    latency_token = _latency_tracker_var.set(latency_tracker)
    recent_turns: list[dict[str, Any]] = []

    llm_provider = active_provider_name()
    mode = "fallback"
    top_score = 0.0
    answer_text = ""
    citations: list[dict[str, Any]] = []
    llm_usage: dict[str, int] = {}
    corrective_metadata: dict[str, Any] = {
        "enabled": bool(settings.corrective_rag_enabled) and not corrective_disabled_for_request(context),
        "attempt_count": 1,
        "query_rewritten": False,
        "correction_reason": None,
        "rewrite_error": None,
        "rewritten_query": None,
    }

    try:
        kb_scope = _resolve_kb_scope(kb_id=kb_id, kb_key=kb_key, auth_context=context.get("auth"))
        context["kb_id"] = kb_scope["id"]
        context["kb_key"] = kb_scope["key"]
        scoped_sid = _scoped_session_id(sid, kb_scope["id"])
        pending = _load_pending_session(scoped_sid)
        recent_turns = _load_recent_turns(scoped_sid, n=settings.conversation_memory_turn_limit)
        if pending and pending.get("pending_clarify_query"):
            merged_query = f"{pending['pending_clarify_query']} | {user_query}"
            prompt_query = merged_query
            followup = True
        else:
            merged_query, resolution_reason = resolve_followup_query(user_query, recent_turns)
            if resolution_reason:
                logger.debug("Resolved follow-up query for session %s using recent history", scoped_sid)

        response_cache_scope = _build_response_cache_scope(
            kb_scope=kb_scope,
            lang=resolved_lang,
            context=context,
            llm_provider=llm_provider,
        )
        response_cache_enabled_for_request = _response_cache_allowed(
            followup=followup,
            recent_turns=recent_turns,
        )
        if response_cache_enabled_for_request:
            response_cache_hit = _get_response_cache_hit(merged_query, response_cache_scope)
            if response_cache_hit is not None:
                payload = response_cache_hit.get("payload") or {}
                cached_answer = str(payload.get("answer_text") or "")
                cached_citations = payload.get("citations") or []
                if cached_answer and isinstance(cached_citations, list):
                    mode = "answer"
                    top_score = float(payload.get("top_score") or 0.0)
                    answer_text = cached_answer
                    citations = cached_citations
                    cache_data = {
                        "response": response_cache_hit.get("type"),
                        "semantic_score": response_cache_hit.get("score"),
                        "cached_query": response_cache_hit.get("query"),
                    }
                    yield {
                        "event": "start",
                        "data": {
                            "query": user_query,
                            "mode": mode,
                            "score": round(top_score, 4),
                            "request_id": context.get("request_id"),
                            "session_id": sid,
                            "llm_provider": llm_provider,
                            "lang": resolved_lang,
                            "kb_id": kb_scope["id"],
                            "kb_key": kb_scope["key"],
                            "kb_name": kb_scope["name"],
                            "kb_version": kb_scope["kb_version"],
                            "cache": cache_data,
                            "corrective_rag": corrective_metadata,
                            "runtime_budget": budget_snapshot(context),
                            "latency_breakdown": latency_tracker.snapshot(),
                        },
                    }
                    yield {"event": "token", "data": {"text": answer_text}}
                    yield {"event": "citations", "data": {"items": citations}}
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    _log_chat(
                        session_id=scoped_sid,
                        user_message=user_query,
                        merged_query=merged_query,
                        mode=mode,
                        top_score=top_score,
                        answer_text=answer_text,
                        citations=citations,
                        latency_ms=latency_ms,
                        llm_provider=llm_provider,
                        request_context=context,
                    )
                    yield {
                        "event": "done",
                        "data": {
                            "ok": True,
                            "latency_ms": latency_ms,
                            "cache": cache_data,
                            "runtime_budget": budget_snapshot(context),
                            "latency_breakdown": latency_tracker.snapshot(),
                        },
                    }
                    return

        results = retrieve(
            merged_query,
            kb_id=kb_scope["id"],
            auth_context=context.get("auth"),
            runtime_context=context,
        )
        results = _apply_lang_boost(results, resolved_lang)
        top_score = float(results[0].get("similarity", 0.0)) if results else 0.0
        results, corrective_metadata = _maybe_corrective_retrieve(
            query=merged_query,
            results=results,
            top_score=top_score,
            lang=resolved_lang,
            kb_id=kb_scope["id"],
            auth_context=context.get("auth"),
            runtime_context=context,
        )
        top_score = float(results[0].get("similarity", 0.0)) if results else 0.0

        mode = "answer" if followup else decide_mode(top_score)
        if mode == "answer" and not results:
            mode = "fallback"

        yield {
            "event": "start",
            "data": {
                "query": user_query,
                "mode": mode,
                "score": round(top_score, 4),
                "request_id": context.get("request_id"),
                "session_id": sid,
                "llm_provider": llm_provider,
                "lang": resolved_lang,
                "kb_id": kb_scope["id"],
                "kb_key": kb_scope["key"],
                "kb_name": kb_scope["name"],
                "kb_version": kb_scope["kb_version"],
                "corrective_rag": corrective_metadata,
                "runtime_budget": budget_snapshot(context),
                "latency_breakdown": latency_tracker.snapshot(),
                "cache": {
                    "response": "miss" if response_cache_enabled_for_request else "disabled",
                },
            },
        }

        if mode == "answer":
            _clear_pending_session(scoped_sid)

            use_llm = settings.normalized_answer_mode != "extractive" and is_llm_ready()
            if use_llm:
                llm_context = _context_for_llm(results)

                # Build prompt with optional conversational memory
                memory = _build_memory_summary(recent_turns)
                full_prompt = (
                    f"{memory}\n\n{CONTEXT_TEMPLATE.format(context=llm_context, question=prompt_query)}"
                    if memory
                    else CONTEXT_TEMPLATE.format(context=llm_context, question=prompt_query)
                )
                answer_text, use_llm, llm_usage = yield from _stream_llm_with_extract_fallback(
                    full_prompt=full_prompt,
                    results=results,
                    lang=resolved_lang,
                )

            else:
                answer_text = _extractive_answer(results, resolved_lang)
                yield {"event": "token", "data": {"text": answer_text}}

            citations = _build_citations(results)

            # Guardrail: if no citations → fallback extractive
            if not citations and use_llm:
                answer_text = _extractive_answer(results, resolved_lang)

            yield {"event": "citations", "data": {"items": citations}}

        elif mode == "clarify":
            # If LLM is available, attempt an answer from partial results first, then invite clarification.
            use_llm = settings.normalized_answer_mode != "extractive" and is_llm_ready()
            if use_llm and results:
                memory = _build_memory_summary(recent_turns)
                full_prompt = (
                    f"{memory}\n\n{CONTEXT_TEMPLATE.format(context=_context_for_llm(results), question=prompt_query)}"
                    if memory
                    else CONTEXT_TEMPLATE.format(context=_context_for_llm(results), question=prompt_query)
                )
                answer_text, use_llm, llm_usage = yield from _stream_llm_with_extract_fallback(
                    full_prompt=full_prompt,
                    results=results,
                    lang=resolved_lang,
                )

                # Append a soft clarification nudge
                nudge = (
                    "\n\n_Nếu thông tin trên chưa đủ, bạn có thể mô tả thêm để mình hỗ trợ chính xác hơn._"
                    if resolved_lang == "vi"
                    else "\n\n_If this doesn't fully answer your question, feel free to give more detail._"
                )
                yield {"event": "token", "data": {"text": nudge}}
                answer_text += nudge
                citations = _build_citations(results)
            else:
                # No LLM — extractive answer + clarification question
                answer_text = _extractive_answer(results, resolved_lang) if results else ""
                if answer_text:
                    yield {"event": "token", "data": {"text": answer_text}}
                clarify_msg = "\n\n" + _clarify_question(results, resolved_lang)
                yield {"event": "token", "data": {"text": clarify_msg}}
                answer_text += clarify_msg
                citations = _build_citations(results)

            category = str(results[0].get("category", "")) if results else ""
            _save_pending_session(scoped_sid, merged_query, category, resolved_lang)
            yield {"event": "citations", "data": {"items": citations}}

        else:
            _clear_pending_session(scoped_sid)
            answer_text = _fallback_text(results, resolved_lang)
            yield {"event": "token", "data": {"text": answer_text}}
            # Even in fallback, show whatever partial citations exist
            citations = _build_citations(results) if results else []
            yield {"event": "citations", "data": {"items": citations}}

        response_cache_store = "skipped"
        if (
            mode == "answer"
            and response_cache_enabled_for_request
            and answer_text
            and citations
        ):
            _set_response_cache(
                merged_query,
                response_cache_scope,
                {
                    "mode": mode,
                    "answer_text": answer_text,
                    "citations": citations,
                    "top_score": top_score,
                    "lang": resolved_lang,
                    "llm_provider": llm_provider,
                    "kb_id": kb_scope["id"],
                    "kb_key": kb_scope["key"],
                    "kb_version": kb_scope["kb_version"],
                    "cached_at": utcnow_iso(),
                },
            )
            response_cache_store = "stored"

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        chat_log_id = _log_chat(
            session_id=scoped_sid,
            user_message=user_query,
            merged_query=merged_query,
            mode=mode,
            top_score=top_score,
            answer_text=answer_text,
            citations=citations,
            latency_ms=latency_ms,
            llm_provider=llm_provider,
            request_context=context,
            llm_usage=llm_usage,
        )
        try:
            record_knowledge_gap(
                chat_log_id=chat_log_id,
                query=user_query,
                mode=mode,
                top_score=top_score,
                session_id=scoped_sid,
                context=context,
            )
        except Exception:
            logger.exception("Failed to record knowledge gap")
        yield {
            "event": "done",
            "data": {
                "ok": True,
                "latency_ms": latency_ms,
                "cache": {
                    "response_store": response_cache_store,
                    "response_cache_enabled": response_cache_enabled_for_request,
                },
                "corrective_rag": corrective_metadata,
                "runtime_budget": budget_snapshot(context),
                "latency_breakdown": latency_tracker.snapshot(),
            },
        }

    except Exception as err:
        logger.error("RAG pipeline error: %s", err, exc_info=True)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        try:
            _log_chat(
                session_id=scoped_sid,
                user_message=user_query,
                merged_query=merged_query,
                mode="error",
                top_score=top_score,
                answer_text=str(err),
                citations=[],
                latency_ms=latency_ms,
                llm_provider=llm_provider,
                request_context=context,
                llm_usage=llm_usage,
            )
        except Exception:
            logger.exception("Failed to store error log")
        yield {"event": "error", "data": {"message": str(err)}}
    finally:
        _latency_tracker_var.reset(latency_token)
