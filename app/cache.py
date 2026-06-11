"""
DiskCache 3-layer caching with TTL and invalidation.
Layers: query embeddings, retrieval results, full responses.
"""

import hashlib
import logging
import os
import time
from typing import Any
from diskcache import Cache
from app.config import settings

logger = logging.getLogger(__name__)

_cache: Cache | None = None


def get_cache() -> Cache:
    """Lazy-init DiskCache."""
    global _cache
    if _cache is None:
        settings.ensure_dirs()
        size_limit = settings.cache_max_size_mb * 1024 * 1024
        _cache = Cache(str(settings.cache_dir), size_limit=size_limit)
        logger.info(f"DiskCache initialized at {settings.cache_dir}")
    return _cache


def _make_key(prefix: str, query: str, scope: str = "") -> str:
    """Create a cache key from prefix + normalized query + scope."""
    raw = f"{query.strip().lower()}:{scope}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}:{h}"


def _make_scope_key(prefix: str, scope: str = "") -> str:
    raw = str(scope or "")
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}:scope:{h}"


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").strip().split())


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for idx, value in enumerate(left):
        current = float(value)
        other = float(right[idx])
        dot += current * other
        left_norm += current * current
        right_norm += other * other
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def _semantic_lookup(
    *,
    prefix: str,
    query_embedding: list[float],
    scope: str,
    threshold: float | None = None,
) -> dict[str, Any] | None:
    cutoff = float(threshold if threshold is not None else settings.semantic_cache_threshold)
    entries = get_cache().get(_make_scope_key(prefix, scope)) or []
    if not isinstance(entries, list):
        return None

    best: dict[str, Any] | None = None
    best_score = 0.0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        score = _cosine_similarity(query_embedding, entry.get("embedding") or [])
        if score > best_score:
            best = entry
            best_score = score

    if best is None or best_score < cutoff:
        return None
    return {
        "query": best.get("query"),
        "score": float(best_score),
        "payload": best.get("payload"),
    }


def _semantic_store(
    *,
    prefix: str,
    query: str,
    query_embedding: list[float],
    scope: str,
    payload: Any,
):
    if not query_embedding:
        return

    normalized = _normalize_query(query)
    key = _make_scope_key(prefix, scope)
    cache = get_cache()
    entries = cache.get(key) or []
    if not isinstance(entries, list):
        entries = []

    next_entry = {
        "query": normalized,
        "embedding": [float(item) for item in query_embedding],
        "payload": payload,
        "cached_at": time.time(),
    }
    deduped = [
        entry
        for entry in entries
        if isinstance(entry, dict) and _normalize_query(str(entry.get("query") or "")) != normalized
    ]
    deduped.append(next_entry)
    max_entries = max(1, int(settings.semantic_cache_max_entries_per_scope))
    cache.set(key, deduped[-max_entries:], expire=settings.cache_ttl_seconds)


# ── Layer 1: Query Embeddings ──────────────────────────────
def get_cached_embedding(query: str) -> list[float] | None:
    key = _make_key("emb", query, settings.effective_embedding_fingerprint)
    return get_cache().get(key)


def set_cached_embedding(query: str, embedding: list[float]):
    key = _make_key("emb", query, settings.effective_embedding_fingerprint)
    get_cache().set(key, embedding, expire=settings.cache_ttl_seconds)


# ── Layer 2: Retrieval Results ─────────────────────────────
def get_cached_retrieval(query: str, scope: str) -> list[dict] | None:
    key = _make_key("ret", query, scope)
    return get_cache().get(key)


def set_cached_retrieval(query: str, scope: str, results: list[dict]):
    key = _make_key("ret", query, scope)
    get_cache().set(key, results, expire=settings.cache_ttl_seconds)


def get_semantic_cached_retrieval(query_embedding: list[float], scope: str) -> dict[str, Any] | None:
    hit = _semantic_lookup(prefix="semret", query_embedding=query_embedding, scope=scope)
    if hit is None:
        return None
    payload = hit.get("payload")
    if not isinstance(payload, list):
        return None
    return {
        "query": hit.get("query"),
        "score": hit.get("score"),
        "results": payload,
    }


def set_semantic_cached_retrieval(query: str, query_embedding: list[float], scope: str, results: list[dict]):
    _semantic_store(
        prefix="semret",
        query=query,
        query_embedding=query_embedding,
        scope=scope,
        payload=results,
    )


# ── Layer 3: Full Responses ───────────────────────────────
def get_cached_response(query: str, scope: str) -> str | None:
    key = _make_key("res", query, scope)
    return get_cache().get(key)


def set_cached_response(query: str, scope: str, response: str):
    key = _make_key("res", query, scope)
    get_cache().set(key, response, expire=settings.cache_ttl_seconds)


def get_cached_response_payload(query: str, scope: str) -> dict[str, Any] | None:
    payload = get_cache().get(_make_key("resp", query, scope))
    return payload if isinstance(payload, dict) else None


def set_cached_response_payload(query: str, scope: str, payload: dict[str, Any]):
    get_cache().set(_make_key("resp", query, scope), payload, expire=settings.cache_ttl_seconds)


def get_semantic_cached_response(query_embedding: list[float], scope: str) -> dict[str, Any] | None:
    hit = _semantic_lookup(prefix="semresp", query_embedding=query_embedding, scope=scope)
    if hit is None:
        return None
    payload = hit.get("payload")
    if not isinstance(payload, dict):
        return None
    return {
        "query": hit.get("query"),
        "score": hit.get("score"),
        "payload": payload,
    }


def set_semantic_cached_response(query: str, query_embedding: list[float], scope: str, payload: dict[str, Any]):
    _semantic_store(
        prefix="semresp",
        query=query,
        query_embedding=query_embedding,
        scope=scope,
        payload=payload,
    )


# ── Admin Operations ──────────────────────────────────────
def clear_cache():
    """Clear all cache entries."""
    cache = get_cache()
    cache.clear()
    logger.info("Cache cleared")


def get_stats() -> dict[str, Any]:
    """Get cache statistics."""
    cache = get_cache()
    volume = cache.volume()
    semantic_retrieval_scopes = 0
    semantic_retrieval_entries = 0
    semantic_response_scopes = 0
    semantic_response_entries = 0
    try:
        for key in cache.iterkeys():
            key_text = str(key)
            if key_text.startswith("semret:scope:"):
                semantic_retrieval_scopes += 1
                entries = cache.get(key) or []
                semantic_retrieval_entries += len(entries) if isinstance(entries, list) else 0
            elif key_text.startswith("semresp:scope:"):
                semantic_response_scopes += 1
                entries = cache.get(key) or []
                semantic_response_entries += len(entries) if isinstance(entries, list) else 0
    except Exception:
        logger.debug("Failed to compute detailed cache stats", exc_info=True)
    return {
        "total_entries": len(cache),
        "size_mb": round(volume / (1024 * 1024), 2),
        "semantic_retrieval_scopes": semantic_retrieval_scopes,
        "semantic_retrieval_entries": semantic_retrieval_entries,
        "semantic_response_scopes": semantic_response_scopes,
        "semantic_response_entries": semantic_response_entries,
    }
