from __future__ import annotations

from types import SimpleNamespace

import app.rag as rag
from app.models import RequestContext
from tests.conftest import configure_test_env, fetch_default_kb


def test_corrective_rag_rewrites_and_retrieves_once_when_first_attempt_has_no_results(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rag.settings, "answer_mode", "extractive")
    monkeypatch.setattr(rag.settings, "corrective_rag_enabled", True)
    monkeypatch.setattr(rag.settings, "corrective_rag_max_attempts", 1)
    monkeypatch.setattr(rag, "is_llm_ready", lambda: True)

    queries: list[str] = []

    def fake_retrieve(query, **kwargs):
        queries.append(query)
        if query == "shipping fee policy":
            return [
                {
                    "text": "Shipping fee is 30000 VND.",
                    "filename": "pricing.csv",
                    "file_type": ".csv",
                    "row_num": 1,
                    "chunk_id": "chunk-shipping-fee",
                    "lang": "en",
                    "similarity": 0.91,
                }
            ]
        return []

    captured: dict[str, object] = {}

    def fake_complete_chat(prompt, system_prompt="", **kwargs):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        captured["timeout_seconds"] = kwargs.get("timeout_seconds")
        captured["max_tokens"] = kwargs.get("max_tokens")
        captured["response_format"] = kwargs.get("response_format")
        return SimpleNamespace(text='{"query":"shipping fee policy"}')

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "complete_chat", fake_complete_chat)

    kb = fetch_default_kb()
    events = list(
        rag.rag_stream(
            query="How much delivery?",
            session_id="phase64-corrective",
            lang="en",
            kb_id=kb.id,
            request_context=RequestContext(request_id="req-phase64-corrective", session_id="phase64-corrective", kb_id=kb.id),
        )
    )

    answer_text = "".join(event["data"]["text"] for event in events if event["event"] == "token")
    start_event = next(event for event in events if event["event"] == "start")
    done_event = next(event for event in events if event["event"] == "done")

    assert queries == ["How much delivery?", "shipping fee policy"]
    assert "Shipping fee is 30000 VND." in answer_text
    assert start_event["data"]["mode"] == "answer"
    assert start_event["data"]["corrective_rag"]["attempt_count"] == 2
    assert start_event["data"]["corrective_rag"]["query_rewritten"] is True
    assert start_event["data"]["corrective_rag"]["correction_reason"] == "no_results"
    assert start_event["data"]["corrective_rag"]["rewritten_query"] == "shipping fee policy"
    assert done_event["data"]["corrective_rag"]["corrected_top_score"] == 0.91
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["timeout_seconds"] == rag.settings.effective_corrective_rag_rewrite_timeout_seconds
    assert captured["max_tokens"] == rag.settings.effective_corrective_rag_rewrite_max_tokens


def test_corrective_rag_does_not_rewrite_high_confidence_results(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rag.settings, "answer_mode", "extractive")
    monkeypatch.setattr(rag.settings, "corrective_rag_enabled", True)
    monkeypatch.setattr(rag, "is_llm_ready", lambda: True)

    def fake_retrieve(query, **kwargs):
        return [
            {
                "text": "Returns are allowed within 7 days.",
                "filename": "returns.csv",
                "file_type": ".csv",
                "row_num": 1,
                "chunk_id": "chunk-returns",
                "lang": "en",
                "similarity": 0.92,
            }
        ]

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "complete_chat", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rewrite not expected")))

    kb = fetch_default_kb()
    events = list(
        rag.rag_stream(
            query="What is the return policy?",
            session_id="phase64-high-confidence",
            lang="en",
            kb_id=kb.id,
            request_context=RequestContext(request_id="req-phase64-high-confidence", session_id="phase64-high-confidence", kb_id=kb.id),
        )
    )

    start_event = next(event for event in events if event["event"] == "start")
    assert start_event["data"]["mode"] == "answer"
    assert start_event["data"]["corrective_rag"]["attempt_count"] == 1
    assert start_event["data"]["corrective_rag"]["query_rewritten"] is False
    assert start_event["data"]["corrective_rag"]["correction_reason"] is None
