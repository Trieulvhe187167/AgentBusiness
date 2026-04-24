from __future__ import annotations

from typing import Any

import httpx

import app.agent as agent
import app.llm_client as llm_client
from app.models import RequestContext
from app.tools import build_default_tool_registry
from tests.conftest import configure_test_env


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_registry_exports_openai_tool_schemas(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    registry = build_default_tool_registry()

    tools = registry.list_openai_tools()
    tool_names = {item["function"]["name"] for item in tools}

    assert tool_names == {
        "search_kb",
        "create_support_ticket",
        "get_order_status",
        "find_recent_orders",
        "get_online_member_count",
        "list_kbs",
        "get_kb_stats",
        "list_google_drive_sources",
        "create_google_drive_source",
        "sync_google_drive_source",
        "get_google_drive_sync_status",
        "delete_google_drive_source",
        "list_support_emails",
        "read_email_thread",
        "create_ticket_from_email",
        "send_email_reply",
    }
    search_tool = next(item for item in tools if item["function"]["name"] == "search_kb")
    assert search_tool["type"] == "function"
    assert search_tool["function"]["parameters"]["type"] == "object"
    assert "query" in search_tool["function"]["parameters"]["properties"]


def test_complete_chat_passes_tools_tool_choice_and_response_format(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_client.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(llm_client.settings, "llm_base_url", "http://localhost:8000/v1")
    monkeypatch.setattr(llm_client.settings, "llm_api_key", "EMPTY")
    monkeypatch.setattr(llm_client.settings, "llm_model", "Qwen/Qwen3-4B-Instruct-2507")

    captured: dict[str, Any] = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "create_support_ticket",
                                        "arguments": '{"issue_type":"shipping","message":"Need help","contact":"user@example.com"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    result = llm_client.complete_chat(
        "Need help with shipping",
        system_prompt="Route the request.",
        tools=[{"type": "function", "function": {"name": "create_support_ticket", "parameters": {"type": "object"}}}],
        tool_choice="auto",
        response_format={"type": "json_object"},
    )

    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["json"]["tools"][0]["function"]["name"] == "create_support_ticket"
    assert captured["json"]["tool_choice"] == "auto"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].name == "create_support_ticket"
    assert result.tool_calls[0].arguments["contact"] == "user@example.com"


def test_decide_route_uses_native_tool_calling_when_enabled(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(agent.settings, "agent_tool_protocol", "openai_tools")
    monkeypatch.setattr(agent.settings, "agent_native_tool_calling", True)

    monkeypatch.setattr(
        agent,
        "complete_chat",
        lambda *args, **kwargs: llm_client.LLMChatResult(
            provider="openai_compatible",
            model="Qwen/Qwen3-4B-Instruct-2507",
            finish_reason="tool_calls",
            tool_calls=[
                llm_client.LLMToolCall(
                    id="call_1",
                    name="create_support_ticket",
                    arguments={
                        "issue_type": "shipping",
                        "message": "Need shipping support",
                        "contact": "user@example.com",
                    },
                    raw_arguments='{"issue_type":"shipping"}',
                )
            ],
        ),
    )

    decision = agent.decide_route(
        "Need shipping support at user@example.com",
        request_context=RequestContext(request_id="req-phase16"),
        lang="en",
    )

    assert decision.route == "tool"
    assert decision.tool_name == "create_support_ticket"
    assert decision.arguments["contact"] == "user@example.com"
    assert decision.reason == "native_tool_call"


def test_decide_route_maps_native_search_kb_tool_call_to_rag(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(agent.settings, "agent_tool_protocol", "openai_tools")
    monkeypatch.setattr(agent.settings, "agent_native_tool_calling", True)

    monkeypatch.setattr(
        agent,
        "complete_chat",
        lambda *args, **kwargs: llm_client.LLMChatResult(
            provider="openai_compatible",
            model="Qwen/Qwen3-4B-Instruct-2507",
            finish_reason="tool_calls",
            tool_calls=[
                llm_client.LLMToolCall(
                    id="call_2",
                    name="search_kb",
                    arguments={"query": "shipping policy", "kb_id": 1},
                    raw_arguments='{"query":"shipping policy","kb_id":1}',
                )
            ],
        ),
    )

    decision = agent.decide_route(
        "What is the shipping policy?",
        request_context=RequestContext(request_id="req-phase16-rag"),
        lang="en",
    )

    assert decision.route == "rag"
    assert decision.tool_name == "search_kb"
    assert decision.arguments["query"] == "shipping policy"
    assert decision.arguments["kb_id"] == 1
