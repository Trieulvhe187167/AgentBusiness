"""
Provider capability metadata for the active LLM runtime.

This keeps system/debug views honest about which features are actually available
for the configured provider and which path the app will use.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config import settings


class ProviderCapabilities(BaseModel):
    provider_config: str
    provider_active: str
    model: str | None = None
    generation_api_surface: str
    router_api_surface: str | None = None
    streaming: bool
    streaming_type: str
    native_tool_calling_supported: bool
    native_tool_calling_enabled: bool
    native_tool_calling_ready: bool
    manual_json_routing_supported: bool = True
    usage_reporting: str
    cached_token_reporting: bool
    prompt_cache_controls: bool
    prompt_cache_key_configured: bool
    prompt_cache_retention: str
    structured_output_supported: bool
    tool_result_continuation: bool
    warnings: list[str] = Field(default_factory=list)


def _generation_surface(provider: str) -> tuple[str, bool, str]:
    if provider == "openai":
        return "responses", True, "sse"
    if provider == "openai_compatible":
        return "chat_completions", True, "sse"
    if provider == "ollama":
        return "ollama_generate", True, "ndjson"
    if provider == "llama_cpp":
        return "llama_cpp_chat", True, "native"
    if provider == "gemini":
        return "gemini_generate_content", False, "chunked_non_streaming"
    if provider == "none":
        return "extractive", False, "none"
    return "unknown", False, "none"


def _usage_reporting(provider: str) -> str:
    if provider == "openai":
        return "full"
    if provider in {"openai_compatible", "gemini", "ollama", "llama_cpp"}:
        return "not_persisted"
    return "none"


def _model_for_provider(provider: str) -> str | None:
    if provider == "openai":
        return settings.openai_model or None
    if provider == "openai_compatible":
        return settings.llm_model or None
    if provider == "gemini":
        return settings.gemini_model or None
    if provider == "ollama":
        return settings.ollama_model or None
    if provider == "llama_cpp":
        return settings.llm_model_path or None
    return None


def resolve_provider_capabilities(*, active_provider: str | None = None) -> ProviderCapabilities:
    configured = settings.normalized_llm_provider
    active = (active_provider or configured or "none").strip().lower()
    model = _model_for_provider(active)
    generation_surface, streaming, streaming_type = _generation_surface(active)

    prompt_cache_retention = settings.normalized_openai_prompt_cache_retention or "api_default"
    prompt_cache_key_configured = bool(settings.openai_prompt_cache_key.strip())
    prompt_cache_controls = active == "openai"
    native_supported = active in {"openai", "openai_compatible"}
    native_enabled = bool(settings.agent_native_tool_calling and settings.normalized_agent_tool_protocol == "openai_tools")

    warnings: list[str] = []
    if configured != active:
        warnings.append(
            f"Configured provider '{configured}' resolved to active provider '{active}'. Check credentials or endpoint availability."
        )
    if prompt_cache_key_configured and active != "openai":
        warnings.append("RAG_OPENAI_PROMPT_CACHE_KEY is configured but only the OpenAI Responses provider can use it.")
    if settings.openai_prompt_cache_retention.strip() and active != "openai":
        warnings.append("RAG_OPENAI_PROMPT_CACHE_RETENTION is configured but only the OpenAI Responses provider can use it.")
    if settings.openai_prompt_cache_retention.strip() and not settings.normalized_openai_prompt_cache_retention:
        warnings.append("RAG_OPENAI_PROMPT_CACHE_RETENTION must be either 'in-memory' or '24h'.")
    if native_enabled and not native_supported:
        warnings.append("Native tool calling is enabled but the active provider does not support the OpenAI tools protocol.")
    if native_enabled and settings.agent_native_tool_status != "ready":
        warnings.append(settings.agent_native_tool_reason)
    if active != "openai":
        warnings.append("Persisted token and cached-token analytics are only available for the OpenAI Responses provider.")
    if active == "gemini":
        warnings.append("Gemini generation currently uses non-streaming generateContent and emits chunked text locally.")

    router_surface = "chat_completions" if native_supported else None
    usage_reporting = _usage_reporting(active)
    return ProviderCapabilities(
        provider_config=configured,
        provider_active=active,
        model=model,
        generation_api_surface=generation_surface,
        router_api_surface=router_surface,
        streaming=streaming,
        streaming_type=streaming_type,
        native_tool_calling_supported=native_supported,
        native_tool_calling_enabled=native_enabled,
        native_tool_calling_ready=bool(native_enabled and settings.agent_native_tool_ready),
        usage_reporting=usage_reporting,
        cached_token_reporting=active == "openai",
        prompt_cache_controls=prompt_cache_controls,
        prompt_cache_key_configured=prompt_cache_key_configured,
        prompt_cache_retention=prompt_cache_retention,
        structured_output_supported=active in {"openai", "openai_compatible"},
        tool_result_continuation=False,
        warnings=warnings,
    )


def provider_capabilities_dict(*, active_provider: str | None = None) -> dict[str, Any]:
    return resolve_provider_capabilities(active_provider=active_provider).model_dump()
