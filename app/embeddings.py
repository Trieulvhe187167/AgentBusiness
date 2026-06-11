"""
Embedding providers.

Default path:
- sentence-transformers local model, with hashing fallback for MVP/offline mode

Optional service paths:
- Hugging Face Text Embeddings Inference style /embed endpoint
- OpenAI-compatible /embeddings endpoint
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import httpx
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

    Priority:
    1. Explicit prefix settings
    2. Explicit instruction settings
    3. Model-family inference for e5/bge
    """
    if settings.embedding_query_prefix or settings.embedding_passage_prefix:
        return settings.embedding_query_prefix, settings.embedding_passage_prefix

    if settings.embedding_query_instruction or settings.embedding_document_instruction:
        query_prefix = (
            f"{settings.embedding_query_instruction.strip()}\nQuery: "
            if settings.embedding_query_instruction.strip()
            else ""
        )
        passage_prefix = (
            f"{settings.embedding_document_instruction.strip()}\nDocument: "
            if settings.embedding_document_instruction.strip()
            else ""
        )
        return query_prefix, passage_prefix

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


def _coerce_embedding_vector(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("embedding vector is not a list")
    return [float(item) for item in value]


def _coerce_embedding_matrix(value: Any, expected_count: int) -> list[list[float]]:
    if isinstance(value, dict):
        if isinstance(value.get("embeddings"), list):
            value = value["embeddings"]
        elif isinstance(value.get("data"), list):
            rows = sorted(
                value["data"],
                key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
            )
            value = [item.get("embedding") for item in rows if isinstance(item, dict)]
    if not isinstance(value, list):
        raise ValueError("embedding response is not a list")
    if len(value) != expected_count:
        raise ValueError(f"embedding response count mismatch: expected={expected_count} actual={len(value)}")
    return [_coerce_embedding_vector(item) for item in value]


def _embedding_base_url() -> str:
    base_url = settings.embedding_base_url.strip().rstrip("/")
    if not base_url:
        raise RuntimeError(f"{settings.normalized_embedding_provider} embedding provider requires RAG_EMBEDDING_BASE_URL")
    return base_url


def _embedding_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.embedding_api_key.strip():
        headers["Authorization"] = f"Bearer {settings.embedding_api_key.strip()}"
    return headers


def _embed_with_tei(texts: list[str]) -> list[list[float]]:
    url = f"{_embedding_base_url()}/embed"
    payload = {"inputs": texts, "normalize": True}
    with httpx.Client(timeout=settings.effective_embedding_timeout_seconds, headers=_embedding_headers()) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return _coerce_embedding_matrix(response.json(), expected_count=len(texts))


def _embed_with_openai_compatible(texts: list[str]) -> list[list[float]]:
    url = f"{_embedding_base_url()}/embeddings"
    payload = {"model": settings.effective_embedding_model_id, "input": texts}
    with httpx.Client(timeout=settings.effective_embedding_timeout_seconds, headers=_embedding_headers()) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return _coerce_embedding_matrix(response.json(), expected_count=len(texts))


def get_model() -> Any | None:
    """Lazy load the local SentenceTransformer model when that provider is active."""
    global _model
    if settings.normalized_embedding_provider != "sentence_transformers":
        return None
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
        kwargs: dict[str, Any] = {}
        if settings.embedding_trust_remote_code:
            kwargs["trust_remote_code"] = True
        _model = SentenceTransformer(source, **kwargs)
        logger.info(
            "Embedding model loaded. backend=sentence-transformers dim=%s fingerprint=%s",
            _model.get_sentence_embedding_dimension(),
            settings.effective_embedding_fingerprint,
        )
        return _model
    except Exception as err:
        logger.warning(
            "Failed to load sentence-transformers model (%s). Falling back to hashing embeddings.",
            err,
        )
        _model = _HASH_SENTINEL
        return None


def warm_up_model() -> None:
    """Eagerly load and probe the active embedding provider."""
    global _embeddings_ready
    logger.info("Embedding warm-up starting...")
    try:
        get_model()
        embed_texts(["warmup"], is_query=True)
        _embeddings_ready = True
        logger.info(
            "Embedding warm-up complete. backend=%s fingerprint=%s",
            embedding_backend_name(),
            settings.effective_embedding_fingerprint,
        )
    except Exception as err:
        logger.error("Embedding warm-up failed: %s", err, exc_info=True)
        _embeddings_ready = False


def is_embeddings_ready() -> bool:
    return _embeddings_ready


def get_dimension() -> int:
    configured = settings.effective_embedding_dimension
    if settings.normalized_embedding_provider != "sentence_transformers":
        if configured:
            return configured
        probe = embed_texts(["dimension probe"], is_query=True)
        if not probe or not probe[0]:
            raise RuntimeError("Remote embedding provider returned an empty dimension probe")
        return len(probe[0])

    model = get_model()
    if model is None:
        return configured or _HASH_DIM
    return int(model.get_sentence_embedding_dimension())


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    if not texts:
        return []

    prepared_texts = _prepare_texts(texts, is_query=is_query)
    provider = settings.normalized_embedding_provider
    if provider == "tei":
        return _embed_with_tei(prepared_texts)
    if provider == "openai_compatible":
        return _embed_with_openai_compatible(prepared_texts)

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
    if settings.normalized_embedding_provider != "sentence_transformers":
        return False
    model = get_model()
    return model is None


def embedding_backend_name() -> str:
    if using_hashing_fallback():
        return "hashing"
    return settings.normalized_embedding_provider
