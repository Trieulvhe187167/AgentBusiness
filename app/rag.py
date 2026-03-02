"""
RAG orchestration with 3-mode logic and persistent clarify sessions.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Generator

from app.cache import (
    get_cached_embedding,
    get_cached_retrieval,
    set_cached_embedding,
    set_cached_retrieval,
)
from app.config import settings
from app.database import execute_sync, fetch_one_sync, utcnow_iso
from app.embeddings import embed_query, using_hashing_fallback
from app.llm_client import active_provider_name, generate_stream, is_llm_ready
from app.models import Citation
from app.vector_store import vector_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a customer support assistant.
Rules:
1) Answer strictly from CONTEXT.
2) If information is missing, say you cannot find it in the provided documents.
3) Keep answers concise and factual.
4) Respond in the user's language.
""".strip()

CONTEXT_TEMPLATE = """--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}
"""

DEFAULT_CLARIFY = {
    "vi": "Bạn có thể cho mình thêm chi tiết để mình trả lời chính xác hơn không?",
    "en": "Could you share a bit more detail so I can answer more precisely?",
}

CLARIFY_BY_CATEGORY = {
    "shipping": {
        "vi": "Bạn ở khu vực nào để mình kiểm tra phí hoặc thời gian giao hàng chính xác?",
        "en": "Which delivery area are you in so I can check the exact fee or timeline?",
    },
    "giao": {
        "vi": "Bạn ở khu vực nào để mình kiểm tra phí hoặc thời gian giao hàng chính xác?",
        "en": "Which delivery area are you in so I can check the exact fee or timeline?",
    },
    "return": {
        "vi": "Bạn muốn đổi sản phẩm hay hoàn tiền, và đơn hàng được mua khi nào?",
        "en": "Do you want an exchange or refund, and when was the order placed?",
    },
    "đổi": {
        "vi": "Bạn muốn đổi sản phẩm hay hoàn tiền, và đơn hàng được mua khi nào?",
        "en": "Do you want an exchange or refund, and when was the order placed?",
    },
    "payment": {
        "vi": "Bạn đang thanh toán theo hình thức nào (COD, chuyển khoản, ví điện tử)?",
        "en": "Which payment method are you using (COD, bank transfer, e-wallet)?",
    },
    "thanh toán": {
        "vi": "Bạn đang thanh toán theo hình thức nào (COD, chuyển khoản, ví điện tử)?",
        "en": "Which payment method are you using (COD, bank transfer, e-wallet)?",
    },
}

FALLBACK_TEXT = {
    "vi": (
        "Mình chưa tìm thấy thông tin này trong tài liệu hiện có. "
        "Bạn có thể hỏi về: {topics}."
    ),
    "en": (
        "I could not find this information in the current documents. "
        "You can ask about: {topics}."
    ),
}


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def detect_language(text: str, explicit_lang: str | None = None) -> str:
    if explicit_lang:
        explicit = explicit_lang.strip().lower()
        if explicit in {"vi", "en"}:
            return explicit

    lowered = text.lower()
    if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", lowered):
        return "vi"

    vi_hints = ["bao nhiêu", "giao hàng", "đổi trả", "thanh toán", "sản phẩm", "khuyến mãi", "ở đâu"]
    if any(hint in lowered for hint in vi_hints):
        return "vi"
    return "en"


def _deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for item in results:
        key = "|".join(
            [
                str(item.get("filename", "")),
                str(item.get("row_num", "")),
                str(item.get("page_num", "")),
                str(item.get("sheet_name", "")),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


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
        row_ref = None
        if item.get("row_num") is not None:
            row_ref = str(item.get("row_num"))

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
        text = text[colon_idx + 2 :].strip()

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


def _fallback_text(results: list[dict[str, Any]], lang: str) -> str:
    categories = []
    seen = set()
    for item in results:
        category = (item.get("category") or "").strip()
        if not category:
            continue
        key = category.lower()
        if key in seen:
            continue
        seen.add(key)
        categories.append(category)

    if not categories:
        categories = ["shipping", "returns", "payment", "products"] if lang == "en" else ["giao hàng", "đổi trả", "thanh toán", "sản phẩm"]

    return FALLBACK_TEXT[lang].format(topics=", ".join(categories[:6]))


def _clarify_question(results: list[dict[str, Any]], lang: str) -> str:
    for item in results:
        category = (item.get("category") or "").lower()
        for key, template in CLARIFY_BY_CATEGORY.items():
            if key in category:
                return template[lang]
    return DEFAULT_CLARIFY[lang]


def _context_for_llm(results: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, item in enumerate(results[: settings.max_answer_chunks], start=1):
        blocks.append(f"[{idx}] Source: {_source_label(item)}\n{item.get('text', '')}")
    return "\n\n".join(blocks)


def decide_mode(top_score: float) -> str:
    threshold_good = settings.threshold_good
    threshold_low = settings.threshold_low
    if using_hashing_fallback():
        threshold_good = min(threshold_good, settings.hashing_threshold_good)
        threshold_low = min(threshold_low, settings.hashing_threshold_low)

    if top_score >= threshold_good:
        return "answer"
    if top_score >= threshold_low:
        return "clarify"
    return "fallback"


def _load_pending_session(session_id: str) -> dict[str, Any] | None:
    return fetch_one_sync("SELECT * FROM chat_sessions WHERE session_id=?", (session_id,))


def _save_pending_session(session_id: str, query: str, category: str, lang: str):
    execute_sync(
        """
        INSERT INTO chat_sessions (
            session_id,
            pending_clarify_query,
            pending_clarify_category,
            pending_clarify_lang,
            updated_at
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
):
    execute_sync(
        """
        INSERT INTO chat_logs (
            session_id,
            user_message,
            merged_query,
            mode,
            top_score,
            answer_text,
            citations_json,
            latency_ms,
            llm_provider,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_message,
            merged_query,
            mode,
            float(top_score),
            answer_text,
            json.dumps(citations, ensure_ascii=False),
            latency_ms,
            llm_provider,
            utcnow_iso(),
        ),
    )


def retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    k = top_k or settings.top_k

    cached_emb = get_cached_embedding(normalized)
    query_embedding = cached_emb if cached_emb is not None else embed_query(normalized)
    if cached_emb is None:
        set_cached_embedding(normalized, query_embedding)

    fingerprint = f"{vector_store.get_index_fingerprint()}:k{k}"
    cached_results = get_cached_retrieval(normalized, fingerprint)
    if cached_results is not None:
        raw = cached_results
    else:
        raw = vector_store.query(query_embedding, top_k=k)
        set_cached_retrieval(normalized, fingerprint, raw)

    floor = settings.min_similarity_threshold
    if using_hashing_fallback():
        floor = min(floor, settings.hashing_min_similarity_threshold)

    filtered = [item for item in (raw or []) if float(item.get("similarity", 0.0)) >= floor]
    return _deduplicate(filtered)


def rag_stream(query: str, session_id: str | None = None, lang: str | None = None) -> Generator[dict[str, Any], None, None]:
    start_time = time.perf_counter()
    user_query = _normalize_query(query)
    resolved_lang = detect_language(user_query, explicit_lang=lang)
    sid = session_id or uuid.uuid4().hex

    merged_query = user_query
    followup = False
    pending = _load_pending_session(sid)
    if pending and pending.get("pending_clarify_query"):
        merged_query = f"{pending['pending_clarify_query']} | {user_query}"
        followup = True

    llm_provider = active_provider_name()
    mode = "fallback"
    top_score = 0.0
    answer_text = ""
    citations: list[dict[str, Any]] = []

    try:
        results = retrieve(merged_query)
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
                "session_id": sid,
                "llm_provider": llm_provider,
                "lang": resolved_lang,
            },
        }

        if mode == "answer":
            _clear_pending_session(sid)

            use_llm = settings.normalized_answer_mode != "extractive" and is_llm_ready()
            if use_llm:
                context = _context_for_llm(results)
                prompt = CONTEXT_TEMPLATE.format(context=context, question=merged_query)
                generated_parts = []
                for token in generate_stream(prompt, system_prompt=SYSTEM_PROMPT):
                    generated_parts.append(token)
                    yield {"event": "token", "data": {"text": token}}
                answer_text = "".join(generated_parts).strip()
            else:
                answer_text = _extractive_answer(results, resolved_lang)
                yield {"event": "token", "data": {"text": answer_text}}

            citations = _build_citations(results)
            yield {"event": "citations", "data": {"items": citations}}

        elif mode == "clarify":
            category = str(results[0].get("category", "")) if results else ""
            _save_pending_session(sid, merged_query, category, resolved_lang)
            answer_text = _clarify_question(results, resolved_lang)
            yield {"event": "token", "data": {"text": answer_text}}
            yield {"event": "citations", "data": {"items": []}}

        else:
            _clear_pending_session(sid)
            answer_text = _fallback_text(results, resolved_lang)
            yield {"event": "token", "data": {"text": answer_text}}
            yield {"event": "citations", "data": {"items": []}}

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        _log_chat(
            session_id=sid,
            user_message=user_query,
            merged_query=merged_query,
            mode=mode,
            top_score=top_score,
            answer_text=answer_text,
            citations=citations,
            latency_ms=latency_ms,
            llm_provider=llm_provider,
        )

        yield {"event": "done", "data": {"ok": True, "latency_ms": latency_ms}}

    except Exception as err:
        logger.error("RAG pipeline error: %s", err, exc_info=True)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        try:
            _log_chat(
                session_id=sid,
                user_message=user_query,
                merged_query=merged_query,
                mode="error",
                top_score=top_score,
                answer_text=str(err),
                citations=[],
                latency_ms=latency_ms,
                llm_provider=llm_provider,
            )
        except Exception:
            logger.exception("Failed to store error log")

        yield {"event": "error", "data": {"message": str(err)}}
