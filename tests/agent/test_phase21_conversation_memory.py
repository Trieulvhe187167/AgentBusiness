from __future__ import annotations

import json

import app.agent as agent
import app.database as database
import app.llm_client as llm_client
import app.rag as rag
from app.models import RequestContext
from tests.conftest import configure_test_env, fetch_default_kb


def _insert_chat_log(session_id: str, *, user_message: str, answer_text: str, mode: str = "answer") -> None:
    database.execute_sync(
        """
        INSERT INTO chat_logs (
            session_id, user_message, merged_query, mode, top_score,
            answer_text, citations_json, latency_ms, llm_provider, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (session_id, user_message, user_message, mode, 0.9, answer_text, "[]", 12, "openai_compatible"),
    )


def test_rag_prompt_includes_only_last_five_session_turns(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rag.settings, "answer_mode", "generative")
    monkeypatch.setattr(rag, "is_llm_ready", lambda: True)
    monkeypatch.setattr(rag, "decide_mode", lambda score: "answer")
    monkeypatch.setattr(rag, "_answer_has_hallucinated_numbers", lambda answer, context: False)

    captured: dict[str, str] = {}

    def fake_generate_stream(prompt: str, system_prompt: str = "", provider: str | None = None):
        captured["prompt"] = prompt
        yield "Gia nay can can nhac."

    monkeypatch.setattr(rag, "generate_stream", fake_generate_stream)
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda query, **kwargs: [
            {
                "text": "San pham A co gia 1.500.000 VND.",
                "filename": "catalog.csv",
                "row_num": 1,
                "category": "product",
                "lang": "vi",
                "similarity": 0.91,
            }
        ],
    )

    kb = fetch_default_kb()
    session_id = "phase21-rag-memory"
    scoped_session_id = rag._scoped_session_id(session_id, kb.id)
    for index in range(1, 7):
        _insert_chat_log(
            scoped_session_id,
            user_message=f"turn user {index}",
            answer_text=f"turn assistant {index}",
        )

    events = list(
        rag.rag_stream(
            query="Dat qua",
            session_id=session_id,
            lang="vi",
            kb_id=kb.id,
            request_context=RequestContext(request_id="req-phase21-rag", session_id=session_id, kb_id=kb.id),
        )
    )

    assert events[-1]["event"] == "done"
    assert "Recent conversation history:" in captured["prompt"]
    assert "turn user 1" not in captured["prompt"]
    assert "turn assistant 1" not in captured["prompt"]
    for index in range(2, 7):
        assert f"turn user {index}" in captured["prompt"]
        assert f"turn assistant {index}" in captured["prompt"]


def test_native_tool_router_receives_last_five_session_turns(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(agent.settings, "agent_tool_protocol", "openai_tools")
    monkeypatch.setattr(agent.settings, "agent_native_tool_calling", True)
    monkeypatch.setattr(agent.settings, "llm_base_url", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(agent.settings, "llm_model", "Qwen/Qwen3-4B-Instruct-2507")

    session_id = "phase21-agent-memory"
    for index in range(1, 7):
        _insert_chat_log(
            session_id,
            user_message=f"agent user {index}",
            answer_text=f"agent assistant {index}",
            mode="tool" if index % 2 == 0 else "answer",
        )

    captured: dict[str, str] = {}

    def fake_complete_chat(*args, **kwargs):
        captured["prompt"] = args[0]
        return llm_client.LLMChatResult(
            provider="openai_compatible",
            model="Qwen/Qwen3-4B-Instruct-2507",
            text='{"route":"rag","message":null,"reason":"conversation_memory_followup"}',
        )

    monkeypatch.setattr(agent, "complete_chat", fake_complete_chat)

    decision = agent.decide_route(
        "Dat qua",
        request_context=RequestContext(request_id="req-phase21-agent", session_id=session_id),
        lang="vi",
    )

    prompt_payload = json.loads(captured["prompt"])

    assert decision.route == "rag"
    assert len(prompt_payload["recent_turns"]) == 5
    assert prompt_payload["recent_turns"][0]["user_message"] == "agent user 2"
    assert prompt_payload["recent_turns"][-1]["user_message"] == "agent user 6"
    assert "agent user 1" not in prompt_payload["conversation_context"]
    assert "agent assistant 1" not in prompt_payload["conversation_context"]
    assert "agent user 6" in prompt_payload["conversation_context"]
    assert "agent assistant 6" in prompt_payload["conversation_context"]


def test_rag_retrieval_resolves_followup_query_from_recent_turns(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rag.settings, "answer_mode", "extractive")
    monkeypatch.setattr(rag, "decide_mode", lambda score: "answer")

    captured: dict[str, str] = {}

    def fake_retrieve(query, **kwargs):
        captured["query"] = query
        return [
            {
                "text": "San pham A co gia 1.500.000 VND.",
                "filename": "catalog.csv",
                "row_num": 1,
                "category": "product",
                "lang": "vi",
                "similarity": 0.91,
            }
        ]

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)

    kb = fetch_default_kb()
    session_id = "phase21-rag-rewrite"
    scoped_session_id = rag._scoped_session_id(session_id, kb.id)
    _insert_chat_log(
        scoped_session_id,
        user_message="Gia cua san pham A la bao nhieu?",
        answer_text="San pham A co gia 1.500.000 VND.",
    )

    events = list(
        rag.rag_stream(
            query="Dat qua",
            session_id=session_id,
            lang="vi",
            kb_id=kb.id,
            request_context=RequestContext(request_id="req-phase21-rag-rewrite", session_id=session_id, kb_id=kb.id),
        )
    )

    assert events[-1]["event"] == "done"
    assert captured["query"].startswith("Gia cua san pham A la bao nhieu?")
    assert "User follow-up about the same topic: Dat qua" in captured["query"]


def test_rag_retrieval_resolves_vietnamese_price_reaction_to_same_topic(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rag.settings, "answer_mode", "extractive")
    monkeypatch.setattr(rag, "decide_mode", lambda score: "answer")

    captured: dict[str, str] = {}

    def fake_retrieve(query, **kwargs):
        captured["query"] = query
        return [
            {
                "text": "Hoc phi la 31.600.000 VND/hoc ky.",
                "filename": "tuition.csv",
                "row_num": 1,
                "category": "admission",
                "lang": "vi",
                "similarity": 0.9,
            }
        ]

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)

    kb = fetch_default_kb()
    session_id = "phase21-rag-price-reaction"
    scoped_session_id = rag._scoped_session_id(session_id, kb.id)
    _insert_chat_log(
        scoped_session_id,
        user_message="Hoc phi",
        answer_text="Hoc phi cho 3 hoc ky dau la 31.600.000 VND/hoc ky.",
    )

    events = list(
        rag.rag_stream(
            query="Gia hoi cao",
            session_id=session_id,
            lang="vi",
            kb_id=kb.id,
            request_context=RequestContext(request_id="req-phase21-price-reaction", session_id=session_id, kb_id=kb.id),
        )
    )

    assert events[-1]["event"] == "done"
    assert captured["query"].startswith("Hoc phi")
    assert "User follow-up about the same topic: Gia hoi cao" in captured["query"]


def test_agent_routes_short_price_reaction_to_memory_with_llm_rewrite(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = fetch_default_kb()
    session_id = "phase21-agent-price-reaction-memory"
    scoped_session_id = rag._scoped_session_id(session_id, kb.id)
    _insert_chat_log(
        scoped_session_id,
        user_message="Hoc phi",
        answer_text="Hoc phi cho 3 hoc ky dau la 31.600.000 VND/hoc ky.",
    )

    captured: dict[str, object] = {}

    def fake_complete_chat(*args, **kwargs):
        captured["prompt"] = args[0]
        captured["system_prompt"] = kwargs.get("system_prompt")
        captured["timeout_seconds"] = kwargs.get("timeout_seconds")
        captured["max_tokens"] = kwargs.get("max_tokens")
        return llm_client.LLMChatResult(provider="ollama", model="llama3.2", text="Nghe cung kha cao do.")

    monkeypatch.setattr(agent, "is_llm_ready", lambda: True)
    monkeypatch.setattr(agent, "complete_chat", fake_complete_chat)

    decision = agent.decide_route(
        "Hoi cao nhi",
        request_context=RequestContext(
            request_id="req-phase21-agent-price-reaction",
            session_id=session_id,
            kb_id=kb.id,
        ),
        lang="vi",
    )

    assert decision.route == "memory"
    assert decision.reason == "reply_from_recent_followup_reaction"
    assert decision.message == "Nghe cung kha cao do."
    assert captured["timeout_seconds"] == agent.settings.agent_followup_reaction_llm_timeout_seconds
    assert captured["max_tokens"] == agent.settings.agent_followup_reaction_llm_max_tokens
    assert '"previous_user": "Hoc phi"' in str(captured["prompt"])


def test_agent_price_reaction_falls_back_to_template_on_llm_failure(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = fetch_default_kb()
    session_id = "phase21-agent-price-reaction-fallback"
    scoped_session_id = rag._scoped_session_id(session_id, kb.id)
    _insert_chat_log(
        scoped_session_id,
        user_message="Hoc phi",
        answer_text="Hoc phi cho 3 hoc ky dau la 31.600.000 VND/hoc ky.",
    )

    monkeypatch.setattr(agent, "is_llm_ready", lambda: True)
    monkeypatch.setattr(
        agent,
        "complete_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(llm_client.LLMTemporaryFailure("timeout")),
    )

    decision = agent.decide_route(
        "Gia hoi cao",
        request_context=RequestContext(
            request_id="req-phase21-agent-price-reaction-fallback",
            session_id=session_id,
            kb_id=kb.id,
        ),
        lang="vi",
    )

    assert decision.route == "memory"
    assert decision.reason == "reply_from_recent_followup_reaction"
    assert "kha cao" in (decision.message or "").lower()


def test_agent_heuristic_route_uses_history_resolved_order_context(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.settings, "agent_native_tool_calling", False)
    monkeypatch.setattr(agent.settings, "agent_tool_protocol", "manual_json")
    monkeypatch.setattr(agent.settings, "llm_provider", "none")

    session_id = "phase21-agent-order-followup"
    _insert_chat_log(
        session_id,
        user_message="Don DH12345 dang giao den dau?",
        answer_text="Don DH12345 dang giao va du kien den trong 2 ngay.",
        mode="tool",
    )

    decision = agent.decide_route(
        "Khi nao toi?",
        request_context=RequestContext(
            request_id="req-phase21-agent-order",
            session_id=session_id,
            auth={"user_id": "user-1", "roles": []},
        ),
        lang="vi",
    )

    assert decision.route == "tool"
    assert decision.tool_name == "get_order_status"
    assert decision.arguments["order_code"] == "DH12345"
