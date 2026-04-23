"""
Embedding wrapper.

Primary path:
- sentence-transformers (if installed)

Fallback path:
- lightweight hashing embeddings (no extra heavy dependencies)
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

_model: Any = None
_HASH_SENTINEL = object()
_HASH_DIM = 384
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Readiness flag — True only after warm_up_model() completes successfully
_embeddings_ready: bool = False


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-10:
        return vec
    return vec / norm


def _hash_embed_text(text: str, dim: int = _HASH_DIM) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return vec.tolist()

    for token in tokens:
        h = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16)
        index = h % dim
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vec[index] += sign

    vec = _normalize(vec)
    return vec.tolist()


def _has_sentence_transformers() -> bool:
    return SentenceTransformer is not None


def _effective_prefixes() -> tuple[str, str]:
    """
    Return (query_prefix, passage_prefix).
    - Explicit config has highest priority.
    - Otherwise infer from model family (e5 / bge).
    """
    if settings.embedding_query_prefix or settings.embedding_passage_prefix:
        return settings.embedding_query_prefix, settings.embedding_passage_prefix

    model_id = settings.effective_embedding_model_id.lower()
    if "e5" in model_id:
        return "query: ", "passage: "
    if "bge" in model_id:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


def _prepare_texts(texts: list[str], is_query: bool) -> list[str]:
    query_prefix, passage_prefix = _effective_prefixes()
    prefix = query_prefix if is_query else passage_prefix
    if not prefix:
        return texts
    return [f"{prefix}{text}" for text in texts]


def get_model() -> Any | None:
    """Lazy load SentenceTransformer model if available."""
    global _model
    if _model is not None:
        return None if _model is _HASH_SENTINEL else _model

    if not _has_sentence_transformers():
        logger.warning(
            "sentence-transformers not installed. Using hashing embedding fallback. "
            "Install sentence-transformers for much better retrieval quality."
        )
        _model = _HASH_SENTINEL
        return None

    source = settings.effective_embedding_source
    if settings.embedding_model_path:
        logger.info("Loading embedding model from local path: %s", source)
    else:
        logger.info("Loading embedding model from HuggingFace: %s", source)

    try:
        _model = SentenceTransformer(source)
        logger.info(
            "Embedding model loaded. backend=sentence-transformers dim=%s model=%s",
            _model.get_sentence_embedding_dimension(),
            source,
        )
        return _model
    except Exception as err:
        logger.warning(
            "Failed to load sentence-transformers model (%s). "
            "Falling back to hashing embeddings.",
            err,
        )
        _model = _HASH_SENTINEL
        return None


def warm_up_model() -> None:
    """
    Eagerly load the model and run a test embed to JIT-warm all internal caches.
    Call this from lifespan via asyncio.to_thread() to avoid blocking the event loop.
    """
    global _embeddings_ready
    logger.info("Embedding warm-up starting...")
    try:
        get_model()                                      # load weights
        embed_texts(["warmup"], is_query=True)           # JIT / cache nóng
        _embeddings_ready = True
        backend = "hashing" if using_hashing_fallback() else "sentence-transformers"
        logger.info("Embedding warm-up complete. backend=%s", backend)
    except Exception as err:
        logger.error("Embedding warm-up failed: %s", err, exc_info=True)
        _embeddings_ready = False


def is_embeddings_ready() -> bool:
    """True only after warm_up_model() has completed (success or partial)."""
    return _embeddings_ready


def get_dimension() -> int:
    model = get_model()
    if model is None:
        return _HASH_DIM
    return int(model.get_sentence_embedding_dimension())


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    if not texts:
        return []

    prepared_texts = _prepare_texts(texts, is_query=is_query)
    model = get_model()
    if model is None:
        return [_hash_embed_text(text) for text in prepared_texts]

    embeddings = model.encode(
        prepared_texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()

    out: list[list[float]] = []
    for emb in embeddings:
        if hasattr(emb, "tolist"):
            out.append(emb.tolist())
        else:
            out.append(list(emb))
    return out


def embed_query(text: str) -> list[float]:
    result = embed_texts([text], is_query=True)
    return result[0] if result else []


def using_hashing_fallback() -> bool:
    model = get_model()
    return model is None
