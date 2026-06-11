"""
Configurable reranking.

Providers:
- none: return retrieval ranking unchanged
- bm25_lite: keyword boost via token overlap with BM25-style scoring
- cross_encoder: neural query-document scoring with BM25-lite fallback
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import re
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ── Stopwords (basic VI + EN) ─────────────────────────────────────────────────
_STOPWORDS: set[str] = {
    # EN
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "for", "by", "with", "from", "and", "or", "but", "not", "no",
    "i", "you", "he", "she", "we", "they", "it", "this", "that", "what",
    "who", "how", "when", "where", "which", "there", "here",
    # VI
    "là", "và", "của", "trong", "có", "không", "được", "cho", "với",
    "về", "tôi", "bạn", "mình", "ở", "thì", "đã", "đang", "sẽ",
    "một", "các", "những", "này", "đó", "khi", "nếu", "vì", "để",
    "như", "hay", "hoặc", "mà", "rằng", "cũng", "vậy", "thế",
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_BM25_K1 = 1.5
_BM25_B = 0.75
_cross_encoder_model: Any = None
_CROSS_ENCODER_MISSING = object()


def _tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    ]


def _bm25_score(query_tokens: list[str], doc_text: str, avg_doc_len: float) -> float:
    """Simple BM25-lite score for a single (query, doc) pair."""
    doc_tokens = _tokenize(doc_text)
    doc_len = len(doc_tokens) or 1
    tf_map: dict[str, int] = {}
    for tok in doc_tokens:
        tf_map[tok] = tf_map.get(tok, 0) + 1

    score = 0.0
    for qt in query_tokens:
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        numerator = tf * (_BM25_K1 + 1)
        denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avg_doc_len)
        score += numerator / denominator

    return score


def _copy_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in results]


def _sort_by_similarity(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
    return results


def _normalize_neural_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= score <= 1.0:
        return round(score, 6)
    # Cross-encoders often return logits. Sigmoid keeps downstream thresholds
    # compatible with the existing 0..1 similarity contract.
    if score >= 50:
        return 1.0
    if score <= -50:
        return 0.0
    return round(1.0 / (1.0 + math.exp(-score)), 6)


def _get_cross_encoder() -> Any | None:
    global _cross_encoder_model
    if _cross_encoder_model is _CROSS_ENCODER_MISSING:
        return None
    if _cross_encoder_model is not None:
        return _cross_encoder_model

    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as err:
        logger.warning("CrossEncoder is unavailable; falling back to BM25-lite reranker: %s", err)
        _cross_encoder_model = _CROSS_ENCODER_MISSING
        return None

    try:
        logger.info("Loading cross-encoder reranker model: %s", settings.reranker_model)
        _cross_encoder_model = CrossEncoder(settings.reranker_model)
        return _cross_encoder_model
    except Exception as err:
        logger.warning(
            "Failed to load cross-encoder reranker model %s; falling back to BM25-lite: %s",
            settings.reranker_model,
            err,
        )
        _cross_encoder_model = _CROSS_ENCODER_MISSING
        return None


def rerank_bm25_lite(
    query: str,
    results: list[dict[str, Any]],
    weight: float | None = None,
) -> list[dict[str, Any]]:
    """
    BM25-lite keyword boost.
    Blends original vector similarity score with BM25 score.
    weight controls how much BM25 contributes.
    Returns results sorted by blended score descending.
    """
    if not results:
        return results

    out = _copy_results(results)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return _sort_by_similarity(out)

    bm25_weight = settings.effective_bm25_reranker_weight if weight is None else max(0.0, min(float(weight), 1.0))
    texts = [item.get("text", "") for item in out]
    avg_len = sum(len(_tokenize(t)) for t in texts) / len(texts) if texts else 1.0

    for item in out:
        bm25 = _bm25_score(query_tokens, item.get("text", ""), avg_len)
        vec_score = float(item.get("similarity", 0.0))
        # Normalize bm25 loosely (cap at 10) then blend
        bm25_norm = min(bm25 / 10.0, 1.0)
        item["retrieval_score"] = round(vec_score, 6)
        item["bm25_score"] = round(bm25, 4)
        item["reranker_provider"] = "bm25_lite"
        item["final_score"] = round((1 - bm25_weight) * vec_score + bm25_weight * bm25_norm, 6)
        item["similarity"] = item["final_score"]

    return _sort_by_similarity(out)


def _predict_cross_encoder(model: Any, pairs: list[tuple[str, str]]) -> list[Any]:
    return list(
        model.predict(
            pairs,
            batch_size=settings.effective_reranker_batch_size,
            show_progress_bar=False,
        )
    )


def _cross_encoder_scores(model: Any, pairs: list[tuple[str, str]]) -> list[Any]:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_predict_cross_encoder, model, pairs)
        return future.result(timeout=settings.effective_reranker_timeout_seconds)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def rerank_cross_encoder(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return results

    model = _get_cross_encoder()
    if model is None:
        return rerank_bm25_lite(query, results)

    candidates = _copy_results(results[: settings.effective_reranker_top_n])
    remainder = _copy_results(results[settings.effective_reranker_top_n:])
    pairs = [(query, str(item.get("text") or "")) for item in candidates]
    try:
        scores = _cross_encoder_scores(model, pairs)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "Cross-encoder reranker timed out after %ss; falling back to BM25-lite",
            settings.effective_reranker_timeout_seconds,
        )
        return rerank_bm25_lite(query, results)
    except Exception as err:
        logger.warning("Cross-encoder reranker failed; falling back to BM25-lite: %s", err)
        return rerank_bm25_lite(query, results)

    if len(scores) != len(candidates):
        logger.warning(
            "Cross-encoder reranker returned %s scores for %s candidates; falling back to BM25-lite",
            len(scores),
            len(candidates),
        )
        return rerank_bm25_lite(query, results)

    weight = settings.effective_reranker_weight
    min_score = max(0.0, min(float(settings.reranker_min_score), 1.0))
    reranked: list[dict[str, Any]] = []
    for item, raw_score in zip(candidates, scores, strict=True):
        retrieval_score = float(item.get("similarity", 0.0))
        reranker_score = _normalize_neural_score(raw_score)
        item["retrieval_score"] = round(retrieval_score, 6)
        item["reranker_score"] = reranker_score
        item["reranker_model"] = settings.reranker_model
        item["reranker_provider"] = "cross_encoder"
        item["final_score"] = round((1.0 - weight) * retrieval_score + weight * reranker_score, 6)
        item["similarity"] = item["final_score"]
        if item["final_score"] >= min_score:
            reranked.append(item)

    reranked = _sort_by_similarity(reranked)
    if remainder:
        reranked.extend(_sort_by_similarity(remainder))
    return reranked


def rerank(
    query: str,
    results: list[dict[str, Any]],
    weight: float | None = None,
) -> list[dict[str, Any]]:
    provider = settings.normalized_reranker_provider
    if provider == "none":
        return _sort_by_similarity(_copy_results(results))
    if provider == "cross_encoder":
        return rerank_cross_encoder(query, results)
    return rerank_bm25_lite(query, results, weight=weight)


def rerank_candidate_limit(final_top_k: int) -> int:
    if settings.normalized_reranker_provider == "cross_encoder":
        return max(int(final_top_k), settings.effective_reranker_top_n)
    return int(final_top_k)
