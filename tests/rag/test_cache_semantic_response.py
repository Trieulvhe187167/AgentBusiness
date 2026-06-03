from __future__ import annotations

import app.rag as rag
from app.config import settings
from tests.conftest import (
    add_vector,
    attach_file,
    configure_test_env,
    fetch_default_kb,
    insert_file,
    mark_ingested,
)


def _patch_common_retrieval(monkeypatch):
    monkeypatch.setattr(rag, "expand_query", lambda query: [query])
    monkeypatch.setattr(rag, "rerank", lambda query, items: items)
    monkeypatch.setattr(rag, "using_hashing_fallback", lambda: False)


def test_semantic_retrieval_cache_reuses_similar_query(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _patch_common_retrieval(monkeypatch)
    monkeypatch.setattr(settings, "semantic_retrieval_cache_enabled", True)
    monkeypatch.setattr(settings, "semantic_cache_threshold", 0.90)

    def _embed(query: str) -> list[float]:
        if "delivery" in query.lower():
            return [0.99, 0.01]
        return [1.0, 0.0]

    monkeypatch.setattr(rag, "embed_query", _embed)
    calls = {"vector": 0}

    def _query(query_embedding, top_k=None, where=None, query_text=None):
        calls["vector"] += 1
        return [
            {
                "chunk_id": "chunk-shipping",
                "text": "Shipping fee is 30000 VND.",
                "similarity": 0.99,
                "filename": "shipping.csv",
                "file_type": ".csv",
            }
        ]

    monkeypatch.setattr(rag.vector_store, "query", _query)

    first = rag._retrieve_single("shipping cost", 5, {}, "scope-a")
    second = rag._retrieve_single("delivery fee", 5, {}, "scope-a")

    assert calls["vector"] == 1
    assert first == second


def test_response_cache_reuses_safe_rag_answer_without_llm_call(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _patch_common_retrieval(monkeypatch)
    monkeypatch.setattr(settings, "response_cache_enabled", True)
    monkeypatch.setattr(settings, "answer_mode", "generative")
    monkeypatch.setattr(rag, "embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr(rag, "is_llm_ready", lambda: True)
    monkeypatch.setattr(rag, "_answer_has_hallucinated_numbers", lambda answer_text, context: False)

    calls = {"llm": 0}

    def _generate_stream(prompt, system_prompt=None):
        calls["llm"] += 1
        return iter(["Shipping fee is 30000 VND."])

    monkeypatch.setattr(rag, "generate_stream", _generate_stream)

    kb = fetch_default_kb()
    file_id = insert_file("response-cache.csv")
    attach_file(kb.id, file_id)
    mark_ingested(kb.id, file_id)
    add_vector(
        kb.id,
        file_id,
        "Shipping fee is 30000 VND.",
        filename="response-cache.csv",
        kb_version=kb.kb_version,
        chunk_id="chunk-response-cache",
    )

    first = list(rag.rag_stream("What is the shipping fee?", session_id="cache-s1", lang="en", kb_id=kb.id))
    second = list(rag.rag_stream("What is the shipping fee?", session_id="cache-s2", lang="en", kb_id=kb.id))

    assert calls["llm"] == 1
    assert [event["event"] for event in first][-1] == "done"
    assert [event["event"] for event in second][-1] == "done"
    second_start = next(event for event in second if event["event"] == "start")
    assert second_start["data"]["cache"]["response"] == "exact"


def test_semantic_response_cache_reuses_paraphrased_safe_answer(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _patch_common_retrieval(monkeypatch)
    monkeypatch.setattr(settings, "response_cache_enabled", True)
    monkeypatch.setattr(settings, "semantic_response_cache_enabled", True)
    monkeypatch.setattr(settings, "semantic_cache_threshold", 0.90)
    monkeypatch.setattr(settings, "answer_mode", "generative")
    monkeypatch.setattr(rag, "is_llm_ready", lambda: True)
    monkeypatch.setattr(rag, "_answer_has_hallucinated_numbers", lambda answer_text, context: False)

    def _embed(query: str) -> list[float]:
        if "delivery" in query.lower():
            return [0.99, 0.01]
        return [1.0, 0.0]

    monkeypatch.setattr(rag, "embed_query", _embed)
    calls = {"llm": 0}

    def _generate_stream(prompt, system_prompt=None):
        calls["llm"] += 1
        return iter(["Shipping fee is 30000 VND."])

    monkeypatch.setattr(rag, "generate_stream", _generate_stream)

    kb = fetch_default_kb()
    file_id = insert_file("semantic-response-cache.csv")
    attach_file(kb.id, file_id)
    mark_ingested(kb.id, file_id)
    add_vector(
        kb.id,
        file_id,
        "Shipping fee is 30000 VND.",
        filename="semantic-response-cache.csv",
        kb_version=kb.kb_version,
        chunk_id="chunk-semantic-response-cache",
    )

    list(rag.rag_stream("shipping cost", session_id="sem-cache-s1", lang="en", kb_id=kb.id))
    second = list(rag.rag_stream("delivery fee", session_id="sem-cache-s2", lang="en", kb_id=kb.id))

    assert calls["llm"] == 1
    second_start = next(event for event in second if event["event"] == "start")
    assert second_start["data"]["cache"]["response"] == "semantic"
    assert second_start["data"]["cache"]["semantic_score"] >= 0.90
