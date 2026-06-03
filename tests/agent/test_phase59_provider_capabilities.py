from __future__ import annotations

from types import SimpleNamespace

import app.main as main
from app.config import settings
from app.provider_capabilities import resolve_provider_capabilities
from tests.conftest import configure_test_env, run


def _request_stub():
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(llm_loaded=False, vector_store_ready=True, embeddings_loaded=False)
        )
    )


def test_openai_capabilities_report_responses_prompt_cache_and_usage(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "openai_prompt_cache_key", "support-rag-v1")
    monkeypatch.setattr(settings, "openai_prompt_cache_retention", "24h")

    cap = resolve_provider_capabilities(active_provider="openai")

    assert cap.provider_active == "openai"
    assert cap.model == "gpt-4o-mini"
    assert cap.generation_api_surface == "responses"
    assert cap.router_api_surface == "chat_completions"
    assert cap.streaming is True
    assert cap.streaming_type == "sse"
    assert cap.usage_reporting == "full"
    assert cap.cached_token_reporting is True
    assert cap.prompt_cache_controls is True
    assert cap.prompt_cache_key_configured is True
    assert cap.prompt_cache_retention == "24h"
    assert cap.structured_output_supported is True
    assert cap.tool_result_continuation is False
    assert cap.warnings == []


def test_openai_compatible_capabilities_report_native_tools_without_cached_tokens(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(settings, "llm_base_url", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(settings, "llm_model", "Qwen/Qwen3-4B-Instruct-2507")
    monkeypatch.setattr(settings, "agent_tool_protocol", "openai_tools")
    monkeypatch.setattr(settings, "agent_native_tool_calling", True)

    cap = resolve_provider_capabilities(active_provider="openai_compatible")

    assert cap.provider_active == "openai_compatible"
    assert cap.model == "Qwen/Qwen3-4B-Instruct-2507"
    assert cap.generation_api_surface == "chat_completions"
    assert cap.streaming is True
    assert cap.native_tool_calling_supported is True
    assert cap.native_tool_calling_enabled is True
    assert cap.native_tool_calling_ready is True
    assert cap.usage_reporting == "not_persisted"
    assert cap.cached_token_reporting is False
    assert cap.prompt_cache_controls is False
    assert "Persisted token and cached-token analytics" in cap.warnings[-1]


def test_capabilities_warn_when_prompt_cache_is_set_for_ollama(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "openai_prompt_cache_key", "unused-cache-key")
    monkeypatch.setattr(settings, "agent_tool_protocol", "openai_tools")
    monkeypatch.setattr(settings, "agent_native_tool_calling", True)

    cap = resolve_provider_capabilities(active_provider="ollama")

    assert cap.provider_active == "ollama"
    assert cap.generation_api_surface == "ollama_generate"
    assert cap.native_tool_calling_supported is False
    assert cap.native_tool_calling_enabled is True
    assert cap.native_tool_calling_ready is False
    assert cap.prompt_cache_controls is False
    assert any("PROMPT_CACHE_KEY" in warning for warning in cap.warnings)
    assert any("Native tool calling is enabled" in warning for warning in cap.warnings)


def test_system_info_includes_llm_capabilities(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_model", "gpt-4o-mini")

    system = run(main.system_info(_request_stub(), kb_id=None, kb_key=None))

    assert system["llm_provider_active"] == "openai"
    assert system["llm_capabilities"]["provider_active"] == "openai"
    assert system["llm_capabilities"]["generation_api_surface"] == "responses"
    assert system["llm_capabilities"]["usage_reporting"] == "full"
