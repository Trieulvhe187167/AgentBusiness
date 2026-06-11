from __future__ import annotations

import concurrent.futures

import app.rag as rag
import app.reranker as reranker
from tests.conftest import configure_test_env, fetch_default_kb


def test_bm25_lite_reranker_preserves_metadata_and_scores(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_provider", "bm25_lite")
    results = [
        {
            "chunk_id": "chunk-a",
            "filename": "a.csv",
            "text": "refund policy allows returns",
            "similarity": 0.50,
            "page_num": 1,
        },
        {
            "chunk_id": "chunk-b",
            "filename": "b.csv",
            "text": "delivery schedule",
            "similarity": 0.30,
            "page_num": 2,
        },
    ]

    reranked = reranker.rerank("refund policy", results)

    assert reranked[0]["chunk_id"] == "chunk-a"
    assert reranked[0]["filename"] == "a.csv"
    assert reranked[0]["page_num"] == 1
    assert reranked[0]["reranker_provider"] == "bm25_lite"
    assert reranked[0]["retrieval_score"] == 0.5
    assert reranked[0]["bm25_score"] > 0
    assert reranked[0]["final_score"] == reranked[0]["similarity"]
    assert "bm25_score" not in results[0]


def test_cross_encoder_reranker_scores_candidates_with_fallback_metadata(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_provider", "cross_encoder")
    monkeypatch.setattr(reranker.settings, "reranker_model", "fake-reranker")
    monkeypatch.setattr(reranker.settings, "reranker_weight", 1.0)
    monkeypatch.setattr(reranker.settings, "reranker_top_n", 10)

    class FakeCrossEncoder:
        def predict(self, pairs, batch_size=8, show_progress_bar=False):
            assert pairs[0][0] == "shipping refund"
            return [-2.0, 3.0]

    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda: FakeCrossEncoder())

    results = [
        {"chunk_id": "low", "filename": "low.csv", "text": "shipping timing", "similarity": 0.9},
        {"chunk_id": "high", "filename": "high.csv", "text": "refund and shipping policy", "similarity": 0.2},
    ]

    reranked = reranker.rerank("shipping refund", results)

    assert [item["chunk_id"] for item in reranked[:2]] == ["high", "low"]
    assert reranked[0]["reranker_provider"] == "cross_encoder"
    assert reranked[0]["reranker_model"] == "fake-reranker"
    assert reranked[0]["reranker_score"] > reranked[1]["reranker_score"]
    assert reranked[0]["retrieval_score"] == 0.2
    assert reranked[0]["final_score"] == reranked[0]["similarity"]
    assert reranked[0]["filename"] == "high.csv"


def test_cross_encoder_reranker_falls_back_to_bm25_on_timeout(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_provider", "cross_encoder")
    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda: object())
    monkeypatch.setattr(
        reranker,
        "_cross_encoder_scores",
        lambda model, pairs: (_ for _ in ()).throw(concurrent.futures.TimeoutError()),
    )

    results = [
        {"chunk_id": "refund", "text": "refund policy", "similarity": 0.4},
        {"chunk_id": "other", "text": "delivery schedule", "similarity": 0.3},
    ]

    reranked = reranker.rerank("refund", results)

    assert reranked[0]["chunk_id"] == "refund"
    assert reranked[0]["reranker_provider"] == "bm25_lite"


def test_retrieve_uses_reranker_candidate_limit_for_cross_encoder(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = fetch_default_kb()
    monkeypatch.setattr(rag.settings, "reranker_provider", "cross_encoder")
    monkeypatch.setattr(rag.settings, "reranker_top_n", 7)
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
                "filename": f"doc-{idx}.csv",
                "row_num": idx,
                "text": f"Candidate {idx}",
                "similarity": 1.0 - (idx * 0.01),
                "kb_id": kb.id,
                "access_level": "public",
            }
            for idx in range(top_k)
        ]

    monkeypatch.setattr(rag, "_retrieve_single", fake_retrieve_single)

    results = rag.retrieve("shipping", top_k=3, kb_id=kb.id)

    assert calls == [7]
    assert len(results) == 3
    assert results[0]["chunk_id"] == "chunk-0"


def test_retrieve_can_diversify_top_results_by_source(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = fetch_default_kb()
    monkeypatch.setattr(rag.settings, "reranker_provider", "bm25_lite")
    monkeypatch.setattr(rag.settings, "retrieval_source_diversification_enabled", True)
    monkeypatch.setattr(rag.settings, "retrieval_source_max_chunks_per_source", 1)
    monkeypatch.setattr(rag.settings, "min_similarity_threshold", 0.0)
    monkeypatch.setattr(rag, "expand_query", lambda query: [query])
    monkeypatch.setattr(rag, "using_hashing_fallback", lambda: False)
    monkeypatch.setattr(rag, "rerank", lambda query, items: items)

    def fake_retrieve_single(query, top_k, where, cache_scope):
        return [
                {
                    "chunk_id": "a-1",
                    "source_id": "source-a",
                    "filename": "a.csv",
                    "row_num": 1,
                    "text": "A first",
                    "similarity": 0.99,
                    "kb_id": kb.id,
                "access_level": "public",
            },
            {
                    "chunk_id": "a-2",
                    "source_id": "source-a",
                    "filename": "a.csv",
                    "row_num": 2,
                    "text": "A second",
                    "similarity": 0.98,
                    "kb_id": kb.id,
                "access_level": "public",
            },
            {
                    "chunk_id": "b-1",
                    "source_id": "source-b",
                    "filename": "b.csv",
                    "row_num": 1,
                    "text": "B first",
                "similarity": 0.97,
                "kb_id": kb.id,
                "access_level": "public",
            },
        ]

    monkeypatch.setattr(rag, "_retrieve_single", fake_retrieve_single)

    results = rag.retrieve("shipping", top_k=2, kb_id=kb.id)

    assert [item["chunk_id"] for item in results] == ["a-1", "b-1"]
    assert all(item["source_diversified"] for item in results)
