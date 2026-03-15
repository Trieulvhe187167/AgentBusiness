from __future__ import annotations

import app.database as database
import app.rag as rag
from tests.conftest import (
    add_vector,
    attach_file,
    configure_test_env,
    fetch_default_kb,
    insert_file,
    mark_ingested,
)


def _patch_retrieval(monkeypatch):
    monkeypatch.setattr(rag, "expand_query", lambda query: [query])
    monkeypatch.setattr(rag, "embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr(rag, "rerank", lambda query, items: items)


def test_rag_stream_keeps_request_context_for_chat_logging(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _patch_retrieval(monkeypatch)
    monkeypatch.setattr(rag, "is_llm_ready", lambda: True)
    monkeypatch.setattr(
        rag,
        "generate_stream",
        lambda prompt, system_prompt=None: iter(["Shipping fee is 30000 VND."]),
    )
    monkeypatch.setattr(rag, "_answer_has_hallucinated_numbers", lambda answer_text, context: False)

    kb = fetch_default_kb()
    file_id = insert_file("phase18.csv")
    attach_file(kb.id, file_id)
    mark_ingested(kb.id, file_id)
    add_vector(
        kb.id,
        file_id,
        "Shipping fee is 30000 VND.",
        filename="phase18.csv",
        kb_version=kb.kb_version,
        chunk_id="chunk-phase18",
    )

    events = list(
        rag.rag_stream(
            query="What is the shipping fee?",
            session_id="phase18-rag",
            lang="en",
            kb_id=kb.id,
            request_context={
                "request_id": "req-phase18",
                "session_id": "phase18-rag",
                "auth": {
                    "user_id": "user-1",
                    "roles": ["member"],
                    "channel": "web",
                },
            },
        )
    )

    event_names = [event["event"] for event in events]
    assert "error" not in event_names
    assert event_names[-1] == "done"

    log_row = database.fetch_one_sync(
        """
        SELECT session_id, request_id, user_id, mode, answer_text
        FROM chat_logs
        WHERE request_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        ("req-phase18",),
    )
    assert log_row == {
        "session_id": f"phase18-rag::kb:{kb.id}",
        "request_id": "req-phase18",
        "user_id": "user-1",
        "mode": "answer",
        "answer_text": "Shipping fee is 30000 VND.",
    }
