"""
Runtime budget and latency controls for RAG deployment profiles.

The controls are intentionally conservative: defaults preserve the current
pipeline, while named deployment profiles can clamp expensive steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings


_PROFILE_CAPS: dict[str, dict[str, int]] = {
    "local_cpu": {
        "max_rerank_candidates": 20,
        "max_answer_chunks": 3,
        "retrieval_budget_ms": 2500,
        "llm_budget_ms": 30000,
    },
    "local_gpu": {
        "max_rerank_candidates": 80,
        "max_answer_chunks": 5,
        "retrieval_budget_ms": 1500,
        "llm_budget_ms": 20000,
    },
    "service": {
        "max_rerank_candidates": 120,
        "max_answer_chunks": 6,
        "retrieval_budget_ms": 1200,
        "llm_budget_ms": 15000,
    },
}


@dataclass
class LatencyTracker:
    embedding_ms: int = 0
    vector_query_ms: int = 0
    reranker_ms: int = 0
    llm_ms: int = 0
    cache_hit: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, key: str, started_at: float, *, cache_hit: bool | None = None) -> int:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        current = int(getattr(self, key, 0))
        setattr(self, key, current + elapsed_ms)
        if cache_hit is not None:
            self.cache_hit = bool(cache_hit)
        return elapsed_ms

    def event(self, name: str, **payload: Any) -> None:
        self.events.append({"name": name, **payload})

    def snapshot(self) -> dict[str, Any]:
        return {
            "embedding_ms": self.embedding_ms,
            "vector_query_ms": self.vector_query_ms,
            "reranker_ms": self.reranker_ms,
            "llm_ms": self.llm_ms,
            "cache_hit": self.cache_hit,
            "events": list(self.events),
        }


def normalized_deployment_profile() -> str:
    profile = settings.deployment_profile.strip().lower()
    return profile if profile in {"custom", "local_cpu", "local_gpu", "service"} else "custom"


def _profile_cap(name: str) -> int:
    profile = normalized_deployment_profile()
    return int(_PROFILE_CAPS.get(profile, {}).get(name, 0))


def _context_controls(context: dict[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {}
    controls = context.get("runtime_controls")
    return controls if isinstance(controls, dict) else {}


def context_bool(context: dict[str, Any] | None, key: str) -> bool:
    value = _context_controls(context).get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def effective_max_rerank_candidates() -> int:
    configured = int(settings.runtime_max_rerank_candidates)
    if configured <= 0:
        configured = settings.effective_reranker_top_n
    configured = max(1, min(configured, 500))
    profile_cap = _profile_cap("max_rerank_candidates")
    return min(configured, profile_cap) if profile_cap > 0 else configured


def effective_max_answer_chunks() -> int:
    configured = int(settings.runtime_max_answer_chunks)
    if configured <= 0:
        configured = settings.max_answer_chunks
    configured = max(1, min(configured, 20))
    profile_cap = _profile_cap("max_answer_chunks")
    return min(configured, profile_cap) if profile_cap > 0 else configured


def effective_retrieval_latency_budget_ms() -> int:
    configured = int(settings.runtime_retrieval_latency_budget_ms)
    if configured > 0:
        return max(100, min(configured, 120000))
    profile_cap = _profile_cap("retrieval_budget_ms")
    return profile_cap if profile_cap > 0 else 0


def effective_llm_latency_budget_ms() -> int:
    configured = int(settings.runtime_llm_latency_budget_ms)
    if configured > 0:
        return max(100, min(configured, 300000))
    profile_cap = _profile_cap("llm_budget_ms")
    return profile_cap if profile_cap > 0 else 0


def reranker_disabled_for_request(context: dict[str, Any] | None = None) -> bool:
    if settings.runtime_disable_reranker:
        return True
    if settings.runtime_disable_neural_reranker and settings.normalized_reranker_provider == "cross_encoder":
        return True
    return context_bool(context, "disable_reranker")


def corrective_disabled_for_request(context: dict[str, Any] | None = None) -> bool:
    return bool(settings.runtime_disable_corrective_rag) or context_bool(context, "disable_corrective_rag")


def budget_snapshot(context: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "deployment_profile": normalized_deployment_profile(),
        "max_rerank_candidates": effective_max_rerank_candidates(),
        "max_answer_chunks": effective_max_answer_chunks(),
        "retrieval_latency_budget_ms": effective_retrieval_latency_budget_ms(),
        "llm_latency_budget_ms": effective_llm_latency_budget_ms(),
        "disable_reranker": bool(settings.runtime_disable_reranker),
        "disable_neural_reranker": bool(settings.runtime_disable_neural_reranker),
        "disable_corrective_rag": bool(settings.runtime_disable_corrective_rag),
        "effective_disable_reranker": reranker_disabled_for_request(context),
        "effective_disable_corrective_rag": corrective_disabled_for_request(context),
        "monitoring_enabled": bool(settings.runtime_monitoring_enabled),
    }
