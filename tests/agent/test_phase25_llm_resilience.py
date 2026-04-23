from __future__ import annotations

from typing import Any

import httpx

import app.llm_client as llm_client
import app.rag as rag
import app.agent as agent
from app.models import RequestContext
from tests.conftest import configure_test_env, fetch_default_kb


def test_generate_stream_falls_back_to_native_ollama_on_local_timeout(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_client.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(llm_client.settings, "llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(llm_client.settings, "llm_model", " llama3.2 ")

    captured: dict[str, Any] = {}

    def fake_httpx_stream(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    def fake_ollama_stream(prompt: str, system_prompt: str = "", *, base_url_override=None, model_override=None):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        captured["base_url_override"] = base_url_override
        captured["model_override"] = model_override
        yield "fallback ok"

    monkeypatch.setattr(httpx, "stream", fake_httpx_stream)
    monkeypatch.setattr(llm_client, "_ollama_stream", fake_ollama_stream)

    answer = "".join(llm_client.generate_stream("hello", system_prompt="route", provider="openai_compatible"))

    assert answer == "fallback ok"
    assert captured["base_url_override"] == "http://localhost:11434"
    assert captured["model_override"] == "llama3.2"


def test_rag_stream_falls_back_to_extractive_when_llm_times_out(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rag.settings, "answer_mode", "generative")
    monkeypatch.setattr(rag, "is_llm_ready", lambda: True)
    monkeypatch.setattr(rag, "decide_mode", lambda score: "answer")
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda query, **kwargs: [
            {
                "text": "Shipping fee is 30000 VND.",
                "filename": "pricing.csv",
                "file_type": ".csv",
                "row_num": 1,
                "chunk_id": "chunk-phase25-timeout",
                "lang": "en",
                "similarity": 0.93,
            }
        ],
    )

    def raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(rag, "generate_stream", raise_timeout)

    kb = fetch_default_kb()
    events = list(
        rag.rag_stream(
            query="What is the shipping fee?",
            session_id="phase25-timeout",
            lang="en",
            kb_id=kb.id,
            request_context=RequestContext(request_id="req-phase25-timeout", session_id="phase25-timeout", kb_id=kb.id),
        )
    )

    event_names = [event["event"] for event in events]
    answer_text = "".join(event["data"]["text"] for event in events if event["event"] == "token")

    assert "error" not in event_names
    assert event_names[-1] == "done"
    assert "Shipping fee is 30000 VND." in answer_text


def test_native_tool_router_falls_back_cleanly_on_temporary_llm_failure(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(agent.settings, "agent_tool_protocol", "openai_tools")
    monkeypatch.setattr(agent.settings, "agent_native_tool_calling", True)
    monkeypatch.setattr(agent, "complete_chat", lambda *args, **kwargs: (_ for _ in ()).throw(llm_client.LLMTemporaryFailure("timeout")))

    decision = agent.decide_route(
        "list kbs",
        request_context=RequestContext(request_id="req-phase25-native-router"),
        lang="en",
    )

    assert decision.route == "tool"
    assert decision.tool_name == "list_kbs"
    assert decision.reason == "admin_list_kbs_intent"


def test_decide_route_skips_llm_router_when_disabled(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.settings, "agent_brain_mode", "hybrid")
    monkeypatch.setattr(agent.settings, "agent_enable_llm_router", False)

    called = {"llm": False}

    def fake_llm_route(*args, **kwargs):
        called["llm"] = True
        return None

    monkeypatch.setattr(agent, "_llm_route", fake_llm_route)

    decision = agent.decide_route(
        "What is the shipping policy?",
        request_context=RequestContext(request_id="req-phase25-fast-router"),
        lang="en",
    )

    assert decision.route == "rag"
    assert decision.reason == "default_rag_route"
    assert called["llm"] is False


def test_decide_route_ai_first_prefers_llm_over_memory_and_heuristic(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.settings, "agent_brain_mode", "ai_first")

    called = {"ai": False, "memory": False, "heuristic": False}

    def fake_ai_first_route(*args, **kwargs):
        called["ai"] = True
        return agent.AgentDecision(route="tool", tool_name="list_kbs", reason="ai_first_route")

    def fake_memory_decision(*args, **kwargs):
        called["memory"] = True
        return agent.AgentDecision(route="memory", message="memory")

    def fake_heuristic_route(*args, **kwargs):
        called["heuristic"] = True
        return agent.AgentDecision(route="rag", reason="heuristic")

    monkeypatch.setattr(agent, "_ai_first_route", fake_ai_first_route)
    monkeypatch.setattr(agent, "_memory_decision", fake_memory_decision)
    monkeypatch.setattr(agent, "_heuristic_route", fake_heuristic_route)

    decision = agent.decide_route(
        "Can you list all KBs?",
        request_context=RequestContext(request_id="req-phase25-ai-first"),
        lang="en",
    )

    assert decision.route == "tool"
    assert decision.tool_name == "list_kbs"
    assert decision.reason == "ai_first_route"
    assert called == {"ai": True, "memory": False, "heuristic": False}


def test_llm_route_prompt_includes_slot_memory(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(agent, "is_llm_ready", lambda: True)

    captured: dict[str, str] = {}

    monkeypatch.setattr(
        agent,
        "_load_session_slots",
        lambda context: {"last_kb_id": 9, "subject_type": "kb", "last_order_code": "DH12345"},
    )

    def fake_generate_stream(prompt: str, system_prompt: str = "", provider: str | None = None, **kwargs):
        captured["prompt"] = prompt
        yield '{"route":"rag","message":null,"reason":"slot_memory_used"}'

    monkeypatch.setattr(agent, "generate_stream", fake_generate_stream)

    decision = agent.decide_route(
        "What about that one?",
        request_context=RequestContext(request_id="req-phase25-slot-memory", session_id="phase25-slot-memory"),
        lang="en",
    )

    assert decision.route == "rag"
    assert '"slot_memory"' in captured["prompt"]
    assert '"last_kb_id": 9' in captured["prompt"]


def test_complete_chat_passes_timeout_and_token_overrides_for_openai_compatible(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_client.settings, "llm_provider", "openai_compatible")

    captured: dict[str, Any] = {}

    def fake_chat_request(**kwargs):
        captured.update(kwargs)
        return llm_client.LLMChatResult(provider="openai_compatible", model="test-model", text="ok")

    monkeypatch.setattr(llm_client, "_chat_completions_request", fake_chat_request)

    result = llm_client.complete_chat(
        "hello",
        system_prompt="route",
        provider="openai_compatible",
        timeout_seconds=7,
        max_tokens=55,
    )

    assert result.text == "ok"
    assert captured["timeout_seconds"] == 7
    assert captured["max_tokens"] == 55
