from __future__ import annotations

import app.database as database
import app.rag as rag
from tests.conftest import configure_test_env, fetch_default_kb


def test_rag_stream_persists_openai_usage_on_chat_log(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rag.settings, "answer_mode", "generative")
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda query, **kwargs: [
            {
                "chunk_id": "chunk-phase58-usage",
                "text": "Shipping fee is 30000 VND.",
                "similarity": 0.99,
                "filename": "phase58-usage.csv",
                "file_type": ".csv",
            }
        ],
    )
    monkeypatch.setattr(rag, "is_llm_ready", lambda: True)
    monkeypatch.setattr(rag, "reset_last_generation_usage", lambda: None)
    monkeypatch.setattr(
        rag,
        "get_last_generation_usage",
        lambda: {
            "input_tokens": 1400,
            "output_tokens": 12,
            "total_tokens": 1412,
            "cached_tokens": 1024,
        },
    )
    monkeypatch.setattr(
        rag,
        "generate_stream",
        lambda prompt, system_prompt=None: iter(["Shipping fee is 30000 VND."]),
    )
    monkeypatch.setattr(rag, "_answer_has_hallucinated_numbers", lambda answer_text, context: False)

    kb = fetch_default_kb()
    events = list(
        rag.rag_stream(
            query="What is the shipping fee?",
            session_id="phase58-usage",
            lang="en",
            kb_id=kb.id,
        )
    )

    assert events[-1]["event"] == "done"
    assert database.fetch_one_sync(
        """
        SELECT llm_input_tokens, llm_output_tokens, llm_total_tokens, llm_cached_tokens
        FROM chat_logs
        ORDER BY id DESC
        LIMIT 1
        """
    ) == {
        "llm_input_tokens": 1400,
        "llm_output_tokens": 12,
        "llm_total_tokens": 1412,
        "llm_cached_tokens": 1024,
    }
