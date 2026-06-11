from __future__ import annotations

import app.rag as rag
from app.runtime_controls import budget_snapshot, effective_max_answer_chunks, effective_max_rerank_candidates
from tests.conftest import configure_test_env, fetch_default_kb


def test_runtime_profile_clamps_rerank_candidates_and_answer_chunks(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = fetch_default_kb()
    monkeypatch.setattr(rag.settings, "deployment_profile", "local_cpu")
    monkeypatch.setattr(rag.settings, "reranker_provider", "cross_encoder")
    monkeypatch.setattr(rag.settings, "reranker_top_n", 80)
    monkeypatch.setattr(rag.settings, "max_answer_chunks", 8)
    monkeypatch.setattr(rag.settings, "min_similarity_threshold", 0.0)
    monkeypatch.setattr(rag, "expand_query", lambda query: [query])
    monkeypatch.setattr(rag, "using_hashing_fallback", lambda: False)
    monkeypatch.setattr(rag, "rerank", lambda query, items: items)

    calls: list[int] = []

    def fake_retrieve_single(query, top_k, where, cache_scope):
        calls.append(top_k)
        return [
            {
                "chunk_id": f"chunk-{idx}",
                "filename": "doc.csv",
                "row_num": idx,
                "source_id": f"source-{idx}",
                "text": f"Candidate {idx}",
                "similarity": 1.0 - (idx * 0.01),
                "kb_id": kb.id,
                "access_level": "public",
            }
            for idx in range(top_k)
        ]

    monkeypatch.setattr(rag, "_retrieve_single", fake_retrieve_single)

    results = rag.retrieve("shipping", top_k=5, kb_id=kb.id)

    assert calls == [20]
    assert len(results) == 5
    assert effective_max_rerank_candidates() == 20
    assert effective_max_answer_chunks() == 3
    assert budget_snapshot()["deployment_profile"] == "local_cpu"


def test_request_can_disable_reranker_and_emit_runtime_metadata(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = fetch_default_kb()
    monkeypatch.setattr(rag.settings, "reranker_provider", "cross_encoder")
    monkeypatch.setattr(rag.settings, "reranker_top_n", 50)
    monkeypatch.setattr(rag.settings, "threshold_good", 0.5)
    monkeypatch.setattr(rag.settings, "answer_mode", "extractive")
    monkeypatch.setattr(rag.settings, "min_similarity_threshold", 0.0)
    monkeypatch.setattr(rag, "expand_query", lambda query: [query])
    monkeypatch.setattr(rag, "using_hashing_fallback", lambda: False)
    monkeypatch.setattr(rag, "rerank", lambda query, items: (_ for _ in ()).throw(AssertionError("rerank disabled")))

    calls: list[int] = []

    def fake_retrieve_single(query, top_k, where, cache_scope):
        calls.append(top_k)
        return [
            {
                "chunk_id": "chunk-shipping",
                "filename": "policy.csv",
                "text": "Shipping is free for standard delivery.",
                "similarity": 0.91,
                "kb_id": kb.id,
                "access_level": "public",
            }
        ]

    monkeypatch.setattr(rag, "_retrieve_single", fake_retrieve_single)

    events = list(
        rag.rag_stream(
            "shipping fee",
            session_id="phase66-runtime",
            kb_id=kb.id,
            request_context={
                "request_id": "req-phase66-runtime",
                "runtime_controls": {"disable_reranker": True},
            },
        )
    )
    start = next(event for event in events if event["event"] == "start")
    done = next(event for event in events if event["event"] == "done")

    assert calls == [rag.settings.top_k]
    assert start["data"]["runtime_budget"]["disable_reranker"] is False
    assert start["data"]["runtime_budget"]["effective_disable_reranker"] is True
    assert start["data"]["latency_breakdown"]["reranker_ms"] == 0
    assert done["data"]["latency_breakdown"]["vector_query_ms"] >= 0
    assert "latency_breakdown" in done["data"]
