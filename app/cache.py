"""
DiskCache 3-layer caching with TTL and invalidation.
Layers: query embeddings, retrieval results, full responses.
"""

import hashlib
import logging
import os
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


def _make_key(prefix: str, query: str, kb_version: str = "") -> str:
    """Create a cache key from prefix + query hash + kb_version."""
    raw = f"{query.strip().lower()}:{kb_version}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}:{h}"


# ── Layer 1: Query Embeddings ──────────────────────────────
def get_cached_embedding(query: str) -> list[float] | None:
    key = _make_key("emb", query, settings.effective_embedding_model_id)
    return get_cache().get(key)


def set_cached_embedding(query: str, embedding: list[float]):
    key = _make_key("emb", query, settings.effective_embedding_model_id)
    get_cache().set(key, embedding, expire=settings.cache_ttl_seconds)


# ── Layer 2: Retrieval Results ─────────────────────────────
def get_cached_retrieval(query: str, kb_version: str) -> list[dict] | None:
    key = _make_key("ret", query, kb_version)
    return get_cache().get(key)


def set_cached_retrieval(query: str, kb_version: str, results: list[dict]):
    key = _make_key("ret", query, kb_version)
    get_cache().set(key, results, expire=settings.cache_ttl_seconds)


# ── Layer 3: Full Responses ───────────────────────────────
def get_cached_response(query: str, kb_version: str) -> str | None:
    key = _make_key("res", query, kb_version)
    return get_cache().get(key)


def set_cached_response(query: str, kb_version: str, response: str):
    key = _make_key("res", query, kb_version)
    get_cache().set(key, response, expire=settings.cache_ttl_seconds)


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
    return {
        "total_entries": len(cache),
        "size_mb": round(volume / (1024 * 1024), 2),
    }
