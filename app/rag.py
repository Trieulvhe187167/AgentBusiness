"""
RAG orchestration with 3-mode logic, multilingual support,
query expansion, BM25 reranking, numeric guardrail, and conversational memory.
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
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.embeddings import embed_query, using_hashing_fallback
from app.lang import detect_language
from app.llm_client import active_provider_name, generate_stream, is_llm_ready
from app.models import Citation
from app.query_expander import expand_query
from app.reranker import rerank
from app.vector_store import vector_store

logger = logging.getLogger(__name__)

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

# Regex to detect numbers in text (for guardrail)
_NUMBER_RE = re.compile(r"\b\d[\d,\.\s]*\d|\b\d\b")


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


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
    for idx, item in enumerate(results[: settings.max_answer_chunks], start=1):
        blocks.append(f"[{idx}] Source: {_source_label(item)}\n{item.get('text', '')}")
    return "\n\n".join(blocks)


# ── Conversational memory (rule-based) ───────────────────────────────────────

def _load_recent_turns(session_id: str, n: int = 3) -> list[dict[str, Any]]:
    """Fetch the last n chat turns for the session."""
    try:
        rows = fetch_all_sync(
            """
            SELECT user_message, answer_text, mode FROM chat_logs
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, n),
        )
        return list(reversed(rows))  # chronological order
    except Exception:
        return []


def _build_memory_summary(turns: list[dict[str, Any]]) -> str:
    """
    Build a 1-2 line rule-based context summary from recent turns.
    Extracts key nouns, categories, and entities without calling an LLM.
    """
    if not turns:
        return ""

    snippets = []
    for turn in turns[-2:]:  # last 2 turns at most
        q = (turn.get("user_message") or "").strip()[:80]
        a = _extract_clean_text(turn.get("answer_text") or "")[:120]
        if q:
            snippets.append(f"Q: {q}")
        if a:
            snippets.append(f"A: {a}")

    if not snippets:
        return ""
    return "Recent conversation context:\n" + "\n".join(snippets)


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
            session_id, user_message, merged_query, mode, top_score,
            answer_text, citations_json, latency_ms, llm_provider, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id, user_message, merged_query, mode,
            float(top_score), answer_text,
            json.dumps(citations, ensure_ascii=False),
            latency_ms, llm_provider, utcnow_iso(),
        ),
    )


# ── Core retrieve ─────────────────────────────────────────────────────────────

def _retrieve_single(query: str, top_k: int) -> list[dict[str, Any]]:
    """Retrieve for a single query string (with embedding cache)."""
    normalized = _normalize_query(query)
    cached_emb = get_cached_embedding(normalized)
    query_embedding = cached_emb if cached_emb is not None else embed_query(normalized)
    if cached_emb is None:
        set_cached_embedding(normalized, query_embedding)

    fingerprint = f"{vector_store.get_index_fingerprint()}:k{top_k}"
    cached_results = get_cached_retrieval(normalized, fingerprint)
    if cached_results is not None:
        return cached_results

    raw = vector_store.query(query_embedding, top_k=top_k)
    set_cached_retrieval(normalized, fingerprint, raw)
    return raw or []


def retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Multi-query retrieval: query original + expanded variants.
    Results are merged, deduplicated, filtered by min_similarity, and reranked.
    """
    k = top_k or settings.top_k
    variants = expand_query(query)

    all_raw: list[dict[str, Any]] = []
    for variant in variants:
        all_raw.extend(_retrieve_single(variant, k))

    floor = settings.min_similarity_threshold
    if using_hashing_fallback():
        floor = min(floor, settings.hashing_min_similarity_threshold)

    filtered = [item for item in all_raw if float(item.get("similarity", 0.0)) >= floor]
    deduped = _deduplicate(filtered)

    # BM25 rerank
    reranked = rerank(query, deduped)
    return reranked[:k]


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
) -> Generator[dict[str, Any], None, None]:
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
        results = _apply_lang_boost(results, resolved_lang)
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

                # Build prompt with optional conversational memory
                recent_turns = _load_recent_turns(sid)
                memory = _build_memory_summary(recent_turns)
                full_prompt = (
                    f"{memory}\n\n{CONTEXT_TEMPLATE.format(context=context, question=merged_query)}"
                    if memory
                    else CONTEXT_TEMPLATE.format(context=context, question=merged_query)
                )

                generated_parts = []
                for token in generate_stream(full_prompt, system_prompt=SYSTEM_PROMPT):
                    generated_parts.append(token)
                    yield {"event": "token", "data": {"text": token}}
                answer_text = "".join(generated_parts).strip()

                # Numeric guardrail: if LLM hallucinated numbers → fallback extractive
                if _answer_has_hallucinated_numbers(answer_text, context):
                    logger.warning("Guardrail triggered: falling back to extractive answer")
                    answer_text = _extractive_answer(results, resolved_lang)
                    yield {"event": "token", "data": {"text": "\n[corrected]\n" + answer_text}}

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
                context = _context_for_llm(results)
                recent_turns = _load_recent_turns(sid)
                memory = _build_memory_summary(recent_turns)
                full_prompt = (
                    f"{memory}\n\n{CONTEXT_TEMPLATE.format(context=context, question=merged_query)}"
                    if memory
                    else CONTEXT_TEMPLATE.format(context=context, question=merged_query)
                )
                generated_parts = []
                for token in generate_stream(full_prompt, system_prompt=SYSTEM_PROMPT):
                    generated_parts.append(token)
                    yield {"event": "token", "data": {"text": token}}
                answer_text = "".join(generated_parts).strip()

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
            _save_pending_session(sid, merged_query, category, resolved_lang)
            yield {"event": "citations", "data": {"items": citations}}

        else:
            _clear_pending_session(sid)
            answer_text = _fallback_text(results, resolved_lang)
            yield {"event": "token", "data": {"text": answer_text}}
            # Even in fallback, show whatever partial citations exist
            citations = _build_citations(results) if results else []
            yield {"event": "citations", "data": {"items": citations}}

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
