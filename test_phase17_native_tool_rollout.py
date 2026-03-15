from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import app.agent as agent
import app.llm_client as llm_client
import app.main as main
from app.config import settings
from app.models import RequestContext
from tests.conftest import configure_test_env, run


def _request_stub():
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(llm_loaded=False, vector_store_ready=True, embeddings_loaded=False)
        )
    )


def test_system_info_reports_native_tool_rollout_ready_state(monkeypatch, tmp_path):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(settings, "llm_model", "Qwen/Qwen3-4B-Instruct-2507")
    monkeypatch.setattr(settings, "llm_base_url", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(settings, "agent_tool_protocol", "openai_tools")
    monkeypatch.setattr(settings, "agent_native_tool_calling", True)
    monkeypatch.setattr(settings, "agent_tool_choice_mode", "required")
    monkeypatch.setattr(settings, "agent_tool_parser", "qwen3_coder")

    system = run(main.system_info(_request_stub(), kb_id=None, kb_key=None))

    assert system["agent_runtime"]["tool_protocol"] == "openai_tools"
    assert system["agent_runtime"]["native_tool_calling"] is True
    assert system["agent_runtime"]["tool_choice_mode"] == "required"
    assert system["agent_runtime"]["native_tool_status"] == "ready"
    assert system["agent_runtime"]["native_tool_ready"] is True
    assert system["agent_runtime"]["native_tool_reason"] == "Runtime is ready for native tool calling."
    assert system["agent_runtime"]["native_tool_warning"] is None
    assert system["agent_runtime"]["tool_parser"] == "qwen3_coder"


def test_system_info_reports_misconfigured_native_tool_runtime(monkeypatch, tmp_path):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "agent_tool_protocol", "openai_tools")
    monkeypatch.setattr(settings, "agent_native_tool_calling", True)

    system = run(main.system_info(_request_stub(), kb_id=None, kb_key=None))

    assert system["agent_runtime"]["native_tool_status"] == "misconfigured"
    assert system["agent_runtime"]["native_tool_ready"] is False
    assert "OpenAI-compatible" in system["agent_runtime"]["native_tool_reason"]


def test_native_tool_route_uses_configured_tool_choice_mode(monkeypatch, tmp_path):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(agent.settings, "agent_tool_protocol", "openai_tools")
    monkeypatch.setattr(agent.settings, "agent_native_tool_calling", True)
    monkeypatch.setattr(agent.settings, "agent_tool_choice_mode", "required")
    monkeypatch.setattr(agent.settings, "llm_base_url", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(agent.settings, "llm_model", "Qwen/Qwen3-4B-Instruct-2507")

    captured: dict[str, Any] = {}

    def fake_complete_chat(*args, **kwargs):
        captured["tool_choice"] = kwargs.get("tool_choice")
        captured["tools"] = kwargs.get("tools")
        return llm_client.LLMChatResult(
            provider="openai_compatible",
            model="Qwen/Qwen3-4B-Instruct-2507",
            finish_reason="tool_calls",
            tool_calls=[
                llm_client.LLMToolCall(
                    id="call_required",
                    name="create_support_ticket",
                    arguments={
                        "issue_type": "shipping",
                        "message": "Need support",
                        "contact": "user@example.com",
                    },
                    raw_arguments='{"issue_type":"shipping"}',
                )
            ],
        )

    monkeypatch.setattr(agent, "complete_chat", fake_complete_chat)

    decision = agent.decide_route(
        "Need shipping support at user@example.com",
        request_context=RequestContext(request_id="req-phase17"),
        lang="en",
    )

    assert captured["tool_choice"] == "required"
    assert captured["tools"]
    assert decision.route == "tool"
    assert decision.tool_name == "create_support_ticket"
