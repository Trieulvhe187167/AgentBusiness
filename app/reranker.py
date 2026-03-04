"""
BM25-lite reranker.
Stage 1 (always active): keyword boost via token overlap with BM25-style scoring.
Stage 2 (optional):      cross-encoder if a model is configured.
"""

from __future__ import annotations

import logging
import re
from typing import Any

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


def rerank(
    query: str,
    results: list[dict[str, Any]],
    weight: float = 0.15,
) -> list[dict[str, Any]]:
    """
    Stage 1: BM25-lite keyword boost.
    Blends original vector similarity score with BM25 score.
    weight controls how much BM25 contributes (default 15%).
    Returns results sorted by blended score descending.
    """
    if not results:
        return results

    query_tokens = _tokenize(query)
    if not query_tokens:
        return results

    texts = [item.get("text", "") for item in results]
    avg_len = sum(len(_tokenize(t)) for t in texts) / len(texts) if texts else 1.0

    for item in results:
        bm25 = _bm25_score(query_tokens, item.get("text", ""), avg_len)
        vec_score = float(item.get("similarity", 0.0))
        # Normalize bm25 loosely (cap at 10) then blend
        bm25_norm = min(bm25 / 10.0, 1.0)
        item["similarity"] = round((1 - weight) * vec_score + weight * bm25_norm, 6)
        item["bm25_score"] = round(bm25, 4)

    results.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
    return results
