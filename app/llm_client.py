"""
LLM provider client with pluggable backends.

Supported providers:
- openai (Responses API)
- gemini (generateContent)
- ollama (local REST)
- llama_cpp (local GGUF via llama-cpp-python)
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any, Callable, Generator
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from app.config import settings
from app.observability import content_attrs, gen_ai_attrs, trace_span

logger = logging.getLogger(__name__)

_llama_model = None
_last_generation_usage: ContextVar[dict[str, int] | None] = ContextVar("last_generation_usage", default=None)


class LLMTemporaryFailure(RuntimeError):
    """Expected transient failure from a configured LLM service."""


class LLMToolCall(BaseModel):
    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""


class LLMResponseToolCall(LLMToolCall):
    call_id: str


class LLMChatResult(BaseModel):
    provider: str
    model: str | None = None
    text: str = ""
    finish_reason: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    raw_message: dict[str, Any] = Field(default_factory=dict)


class LLMResponseResult(BaseModel):
    provider: str = "openai"
    model: str | None = None
    response_id: str | None = None
    text: str = ""
    tool_calls: list[LLMResponseToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


def _normalized_model_name(value: str | None) -> str:
    return str(value or "").strip()


def _looks_like_local_ollama_openai_base(base_url: str) -> bool:
    parsed = urlparse((base_url or "").strip())
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1"} and (parsed.port or 80) == 11434


def _resolved_ollama_base_url() -> str:
    configured = settings.effective_ollama_base_url.strip()
    if configured:
        return configured.rstrip("/")
    base = settings.llm_base_url.strip().rstrip("/")
    if _looks_like_local_ollama_openai_base(base):
        return base[:-3] if base.endswith("/v1") else base
    return ""


def _is_transient_httpx_error(err: Exception) -> bool:
    return isinstance(err, (httpx.TimeoutException, httpx.ConnectError))


def _temporary_failure(message: str, err: Exception) -> LLMTemporaryFailure:
    return LLMTemporaryFailure(f"{message}: {err.__class__.__name__}")


def _safe_chunk_text(text: str, chunk_size: int = 24) -> Generator[str, None, None]:
    """Yield short chunks to keep SSE UX smooth even for non-streaming APIs."""
    if not text:
        return
    words = text.split()
    current = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            yield " ".join(current) + " "
            current = []
            current_len = 0
    if current:
        yield " ".join(current)


def _load_llama_cpp():
    global _llama_model
    if _llama_model is not None:
        return _llama_model

    if not settings.llm_model_path:
        raise RuntimeError("RAG_LLM_MODEL_PATH is not configured")

    from llama_cpp import Llama

    logger.info(
        "Loading llama.cpp model from %s (ctx=%s threads=%s)",
        settings.llm_model_path,
        settings.llm_n_ctx,
        settings.effective_threads,
    )
    _llama_model = Llama(
        model_path=settings.llm_model_path,
        n_ctx=settings.llm_n_ctx,
        n_threads=settings.effective_threads,
        n_batch=settings.llm_n_batch,
        verbose=False,
    )
    return _llama_model


def _llama_cpp_stream(
    prompt: str,
    system_prompt: str = "",
    *,
    max_tokens_override: int | None = None,
) -> Generator[str, None, None]:
    model = _load_llama_cpp()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    stream = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens_override or settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        repeat_penalty=settings.llm_repeat_penalty,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        token = delta.get("content", "")
        if token:
            yield token


def _ollama_stream(
    prompt: str,
    system_prompt: str = "",
    *,
    base_url_override: str | None = None,
    model_override: str | None = None,
    timeout_seconds_override: int | None = None,
    max_tokens_override: int | None = None,
) -> Generator[str, None, None]:
    base = (base_url_override or _resolved_ollama_base_url()).rstrip("/")
    if not base:
        raise RuntimeError("RAG_OLLAMA_BASE_URL is not configured")

    url = f"{base}/api/generate"
    model_name = _normalized_model_name(model_override) or _normalized_model_name(settings.ollama_model)
    if not model_name:
        raise RuntimeError("RAG_OLLAMA_MODEL is not configured")
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "temperature": settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "repeat_penalty": settings.llm_repeat_penalty,
            "num_predict": max_tokens_override or settings.llm_max_tokens,
        },
    }

    timeout = httpx.Timeout(timeout_seconds_override or settings.ollama_timeout_seconds)
    try:
        with httpx.stream("POST", url, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
    except Exception as err:
        if _is_transient_httpx_error(err):
            raise _temporary_failure("Native Ollama request failed", err) from err
        raise


def _extract_openai_text(payload: dict) -> str:
    if payload.get("output_text"):
        return payload["output_text"]

    chunks = []
    for out in payload.get("output", []):
        for content in out.get("content", []):
            text = content.get("text")
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def _extract_openai_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cached_tokens": int(input_details.get("cached_tokens") or 0),
    }


def combine_usage(*items: dict[str, int] | None) -> dict[str, int]:
    combined = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    for item in items:
        for key in combined:
            combined[key] += int((item or {}).get(key) or 0)
    return combined


def _set_usage_span_attrs(span: Any, usage: dict[str, int]) -> None:
    span.set_attribute("gen_ai.usage.input_tokens", int(usage.get("input_tokens") or 0))
    span.set_attribute("gen_ai.usage.output_tokens", int(usage.get("output_tokens") or 0))
    span.set_attribute("gen_ai.usage.total_tokens", int(usage.get("total_tokens") or 0))
    span.set_attribute("gen_ai.usage.cached_tokens", int(usage.get("cached_tokens") or 0))


def reset_last_generation_usage() -> None:
    _last_generation_usage.set(None)


def get_last_generation_usage() -> dict[str, int]:
    return dict(_last_generation_usage.get() or {})


def _record_generation_usage(span: Any, usage: dict[str, int]) -> None:
    normalized = {key: int(value or 0) for key, value in usage.items()}
    _last_generation_usage.set(normalized)
    _set_usage_span_attrs(span, normalized)


def _coerce_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    chunks.append(str(text))
            elif item:
                chunks.append(str(item))
        return "\n".join(chunks).strip()
    return ""


def _parse_tool_calls(message: dict[str, Any]) -> list[LLMToolCall]:
    parsed: list[LLMToolCall] = []
    for item in message.get("tool_calls") or []:
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_arguments = str(function.get("arguments") or "")
        arguments: dict[str, Any] = {}
        if raw_arguments:
            try:
                payload = json.loads(raw_arguments)
                if isinstance(payload, dict):
                    arguments = payload
            except json.JSONDecodeError:
                logger.warning("Failed to decode tool arguments for %s: %s", name, raw_arguments[:160])
        parsed.append(
            LLMToolCall(
                id=str(item.get("id")) if item.get("id") is not None else None,
                name=name,
                arguments=arguments,
                raw_arguments=raw_arguments,
            )
        )
    return parsed


def _build_chat_messages(prompt: str, system_prompt: str = "") -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _chat_completions_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str = "",
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    timeout_seconds: int,
    max_tokens: int | None = None,
) -> LLMChatResult:
    base = base_url.rstrip("/")
    if not base:
        raise RuntimeError("Chat completions base URL is not configured")

    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.upper() != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    body: dict[str, Any] = {
        "model": _normalized_model_name(model),
        "messages": _build_chat_messages(prompt, system_prompt),
        "temperature": settings.llm_temperature,
        "top_p": settings.llm_top_p,
        "frequency_penalty": max(0.0, settings.llm_repeat_penalty - 1.0),
        "max_tokens": max_tokens or settings.llm_max_tokens,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if response_format is not None:
        body["response_format"] = response_format

    timeout = httpx.Timeout(timeout_seconds)
    try:
        response = httpx.post(url, headers=headers, json=body, timeout=timeout)
    except Exception as err:
        if _is_transient_httpx_error(err):
            raise _temporary_failure("Chat completions request failed", err) from err
        raise
    response.raise_for_status()
    payload = response.json()
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return LLMChatResult(
        provider="openai_compatible",
        model=payload.get("model"),
        text=_coerce_message_text(message.get("content")),
        finish_reason=choice.get("finish_reason"),
        tool_calls=_parse_tool_calls(message),
        raw_message=message if isinstance(message, dict) else {},
    )


def _openai_stream(
    prompt: str,
    system_prompt: str = "",
    *,
    timeout_seconds_override: int | None = None,
    max_tokens_override: int | None = None,
    on_completed: Callable[[dict[str, int]], None] | None = None,
) -> Generator[str, None, None]:
    if not settings.openai_api_key:
        raise RuntimeError("RAG_OPENAI_API_KEY is not configured")

    base = settings.openai_base_url.rstrip("/")
    url = f"{base}/responses"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    user_input = [{"type": "input_text", "text": prompt}]
    input_items = [{"role": "user", "content": user_input}]
    if system_prompt:
        input_items.insert(0, {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]})

    body = {
        "model": _normalized_model_name(settings.openai_model),
        "input": input_items,
        "temperature": settings.llm_temperature,
        "top_p": settings.llm_top_p,
        "frequency_penalty": max(0.0, settings.llm_repeat_penalty - 1.0),
        "max_output_tokens": max_tokens_override or settings.llm_max_tokens,
        "stream": True,
    }
    prompt_cache_key = settings.openai_prompt_cache_key.strip()
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key
    prompt_cache_retention = settings.normalized_openai_prompt_cache_retention
    if prompt_cache_retention:
        body["prompt_cache_retention"] = prompt_cache_retention

    timeout = httpx.Timeout(timeout_seconds_override or settings.openai_timeout_seconds)
    try:
        with httpx.stream("POST", url, headers=headers, json=body, timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.debug("Ignored malformed OpenAI Responses SSE event: %s", data_str[:160])
                    continue
                event_type = str(event.get("type") or "")
                if event_type == "response.output_text.delta":
                    delta = str(event.get("delta") or "")
                    if delta:
                        yield delta
                elif event_type == "response.completed":
                    usage = _extract_openai_usage(event.get("response") or {})
                    if on_completed:
                        on_completed(usage)
                    logger.info(
                        "OpenAI Responses usage: input_tokens=%s output_tokens=%s total_tokens=%s cached_tokens=%s",
                        usage["input_tokens"],
                        usage["output_tokens"],
                        usage["total_tokens"],
                        usage["cached_tokens"],
                    )
    except Exception as err:
        if _is_transient_httpx_error(err):
            raise _temporary_failure("OpenAI Responses request failed", err) from err
        raise


def _parse_response_tool_calls(payload: dict[str, Any]) -> list[LLMResponseToolCall]:
    parsed: list[LLMResponseToolCall] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        name = str(item.get("name") or "").strip()
        call_id = str(item.get("call_id") or item.get("id") or "").strip()
        raw_arguments = str(item.get("arguments") or "")
        if not name or not call_id:
            continue
        arguments: dict[str, Any] = {}
        if raw_arguments:
            try:
                decoded = json.loads(raw_arguments)
                if isinstance(decoded, dict):
                    arguments = decoded
            except json.JSONDecodeError:
                logger.warning("Failed to decode Responses tool arguments for %s: %s", name, raw_arguments[:160])
        parsed.append(
            LLMResponseToolCall(
                id=str(item.get("id")) if item.get("id") is not None else None,
                call_id=call_id,
                name=name,
                arguments=arguments,
                raw_arguments=raw_arguments,
            )
        )
    return parsed


def _openai_response_body(
    *,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    previous_response_id: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": _normalized_model_name(settings.openai_model),
        "input": input_items,
        "temperature": settings.llm_temperature,
        "top_p": settings.llm_top_p,
        "frequency_penalty": max(0.0, settings.llm_repeat_penalty - 1.0),
        "max_output_tokens": max_tokens or settings.llm_max_tokens,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    prompt_cache_key = settings.openai_prompt_cache_key.strip()
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key
    prompt_cache_retention = settings.normalized_openai_prompt_cache_retention
    if prompt_cache_retention:
        body["prompt_cache_retention"] = prompt_cache_retention
    return body


def openai_create_response(
    *,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    previous_response_id: str | None = None,
    timeout_seconds: int | None = None,
    max_tokens: int | None = None,
) -> LLMResponseResult:
    if not settings.openai_api_key:
        raise RuntimeError("RAG_OPENAI_API_KEY is not configured")

    base = settings.openai_base_url.rstrip("/")
    url = f"{base}/responses"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    body = _openai_response_body(
        input_items=input_items,
        tools=tools,
        previous_response_id=previous_response_id,
        max_tokens=max_tokens,
    )
    timeout = httpx.Timeout(timeout_seconds or settings.openai_timeout_seconds)
    with trace_span(
        "gen_ai.responses",
        {
            **gen_ai_attrs(
                operation="responses",
                system="openai",
                request_model=settings.openai_model,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
            ),
            "gen_ai.request.tool_count": len(tools or []),
            "gen_ai.request.previous_response_id": previous_response_id or "",
        },
    ) as span:
        try:
            response = httpx.post(url, headers=headers, json=body, timeout=timeout)
        except Exception as err:
            if _is_transient_httpx_error(err):
                raise _temporary_failure("OpenAI Responses request failed", err) from err
            raise
        response.raise_for_status()
        payload = response.json()
        usage = _extract_openai_usage(payload)
        _set_usage_span_attrs(span, usage)
        result = LLMResponseResult(
            model=payload.get("model") or settings.openai_model,
            response_id=payload.get("id"),
            text=_extract_openai_text(payload),
            tool_calls=_parse_response_tool_calls(payload),
            usage=usage,
            raw=payload,
        )
        span.set_attribute("gen_ai.response.id", result.response_id or "")
        span.set_attribute("gen_ai.response.tool_call_count", len(result.tool_calls))
        span.set_attribute("gen_ai.response.text_length", len(result.text or ""))
        return result


def openai_continue_response_with_tool_outputs(
    *,
    previous_response_id: str,
    tool_outputs: list[dict[str, str]],
    timeout_seconds: int | None = None,
    max_tokens: int | None = None,
) -> LLMResponseResult:
    input_items = [
        {
            "type": "function_call_output",
            "call_id": str(item["call_id"]),
            "output": str(item.get("output") or ""),
        }
        for item in tool_outputs
        if item.get("call_id")
    ]
    return openai_create_response(
        input_items=input_items,
        previous_response_id=previous_response_id,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
    )


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    return text.strip()


def _gemini_stream(
    prompt: str,
    system_prompt: str = "",
    *,
    timeout_seconds_override: int | None = None,
    max_tokens_override: int | None = None,
) -> Generator[str, None, None]:
    if not settings.gemini_api_key:
        raise RuntimeError("RAG_GEMINI_API_KEY is not configured")

    base = settings.gemini_base_url.rstrip("/")
    model = _normalized_model_name(settings.gemini_model)
    url = f"{base}/models/{model}:generateContent?key={settings.gemini_api_key}"

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": settings.llm_temperature,
            "topP": settings.llm_top_p,
            "maxOutputTokens": max_tokens_override or settings.llm_max_tokens,
        },
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    timeout = httpx.Timeout(timeout_seconds_override or settings.gemini_timeout_seconds)
    try:
        response = httpx.post(url, json=body, timeout=timeout)
    except Exception as err:
        if _is_transient_httpx_error(err):
            raise _temporary_failure("Gemini request failed", err) from err
        raise
    response.raise_for_status()
    text = _extract_gemini_text(response.json())
    for chunk in _safe_chunk_text(text):
        yield chunk


def _openai_compatible_stream(
    prompt: str,
    system_prompt: str = "",
    *,
    timeout_seconds_override: int | None = None,
    max_tokens_override: int | None = None,
) -> Generator[str, None, None]:
    base = settings.llm_base_url.rstrip("/")
    if not base:
        raise RuntimeError("RAG_LLM_BASE_URL is not configured for openai_compatible")

    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key and settings.llm_api_key.upper() != "EMPTY":
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": _normalized_model_name(settings.llm_model),
        "messages": messages,
        "temperature": settings.llm_temperature,
        "top_p": settings.llm_top_p,
        "frequency_penalty": max(0.0, settings.llm_repeat_penalty - 1.0),
        "max_tokens": max_tokens_override or settings.llm_max_tokens,
        "stream": True,
    }

    timeout = httpx.Timeout(timeout_seconds_override or settings.llm_timeout_seconds)
    try:
        with httpx.stream("POST", url, headers=headers, json=body, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                    except json.JSONDecodeError:
                        continue
    except (httpx.TimeoutException, httpx.ConnectError) as err:
        if not _looks_like_local_ollama_openai_base(base):
            raise _temporary_failure("OpenAI-compatible stream request failed", err) from err
        logger.warning(
            "OpenAI-compatible request to %s failed (%s). Falling back to native Ollama API.",
            base,
            err.__class__.__name__,
        )
        try:
            ollama_kwargs: dict[str, Any] = {
                "base_url_override": _resolved_ollama_base_url(),
                "model_override": _normalized_model_name(settings.llm_model),
            }
            if timeout_seconds_override is not None:
                ollama_kwargs["timeout_seconds_override"] = timeout_seconds_override
            if max_tokens_override is not None:
                ollama_kwargs["max_tokens_override"] = max_tokens_override
            try:
                yield from _ollama_stream(prompt, system_prompt, **ollama_kwargs)
            except TypeError:
                # Keep older tests and monkeypatched helpers working when they do not accept
                # the newer override kwargs.
                yield from _ollama_stream(
                    prompt,
                    system_prompt,
                    base_url_override=ollama_kwargs["base_url_override"],
                    model_override=ollama_kwargs["model_override"],
                )
        except LLMTemporaryFailure as fallback_err:
            raise _temporary_failure("Local Ollama fallback failed", fallback_err) from fallback_err


def complete_chat(
    prompt: str,
    system_prompt: str = "",
    *,
    provider: str | None = None,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
    max_tokens: int | None = None,
) -> LLMChatResult:
    selected = (provider or choose_provider()).lower().strip()
    model_override = _normalized_model_name(model)
    request_model = model_override or settings.effective_chat_model or None
    with trace_span(
        "gen_ai.chat",
        {
            **gen_ai_attrs(
                operation="chat",
                system=selected,
                request_model=request_model,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
            ),
            "gen_ai.request.tool_count": len(tools or []),
            **content_attrs("gen_ai.prompt", prompt),
        },
    ) as span:
        if selected == "openai_compatible":
            result = _chat_completions_request(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=model_override or _normalized_model_name(settings.llm_model),
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                timeout_seconds=timeout_seconds or settings.llm_timeout_seconds,
                max_tokens=max_tokens,
            )
        elif selected == "openai":
            result = _chat_completions_request(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
                model=model_override or _normalized_model_name(settings.openai_model),
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                timeout_seconds=timeout_seconds or settings.openai_timeout_seconds,
                max_tokens=max_tokens,
            ).model_copy(update={"provider": "openai"})
        else:
            text = "".join(
                generate_stream(
                    prompt,
                    system_prompt=system_prompt,
                    provider=selected,
                    timeout_seconds=timeout_seconds,
                    max_tokens=max_tokens,
                )
            ).strip()
            result = LLMChatResult(provider=selected, model=settings.effective_chat_model or None, text=text)
        span.set_attribute("gen_ai.response.model", result.model or request_model or "")
        span.set_attribute("gen_ai.response.finish_reasons", [result.finish_reason] if result.finish_reason else [])
        span.set_attribute("gen_ai.response.tool_call_count", len(result.tool_calls))
        span.set_attribute("gen_ai.response.text_length", len(result.text or ""))
        return result


def provider_available(provider: str) -> bool:
    provider = provider.lower().strip()
    if provider == "openai":
        return bool(settings.openai_api_key)
    if provider == "openai_compatible":
        return bool(settings.llm_base_url)
    if provider == "gemini":
        return bool(settings.gemini_api_key)
    if provider == "ollama":
        return bool(settings.effective_ollama_base_url)
    if provider == "llama_cpp":
        return bool(settings.llm_model_path)
    if provider in {"none", "extractive"}:
        return True
    return False


def choose_provider() -> str:
    configured = settings.normalized_llm_provider
    if configured != "auto":
        return configured if provider_available(configured) else "none"

    for candidate in ("openai", "gemini", "ollama", "llama_cpp", "openai_compatible"):
        if provider_available(candidate):
            return candidate
    return "none"


def generate_stream(
    prompt: str,
    system_prompt: str = "",
    provider: str | None = None,
    *,
    timeout_seconds: int | None = None,
    max_tokens: int | None = None,
) -> Generator[str, None, None]:
    selected = (provider or choose_provider()).lower().strip()
    reset_last_generation_usage()
    with trace_span(
        "gen_ai.generate_stream",
        {
            **gen_ai_attrs(
                operation="chat_stream",
                system=selected,
                request_model=settings.effective_chat_model or None,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
            ),
            **content_attrs("gen_ai.prompt", prompt),
        },
    ) as span:
        token_count = 0
        text_length = 0
        try:
            if selected == "openai":
                span.set_attribute("gen_ai.request.prompt_cache_key_configured", bool(settings.openai_prompt_cache_key.strip()))
                span.set_attribute(
                    "gen_ai.request.prompt_cache_retention",
                    settings.normalized_openai_prompt_cache_retention or "api_default",
                )
                stream = _openai_stream(
                    prompt,
                    system_prompt,
                    timeout_seconds_override=timeout_seconds,
                    max_tokens_override=max_tokens,
                    on_completed=lambda usage: _record_generation_usage(span, usage),
                )
            elif selected == "openai_compatible":
                stream = _openai_compatible_stream(
                    prompt,
                    system_prompt,
                    timeout_seconds_override=timeout_seconds,
                    max_tokens_override=max_tokens,
                )
            elif selected == "gemini":
                stream = _gemini_stream(
                    prompt,
                    system_prompt,
                    timeout_seconds_override=timeout_seconds,
                    max_tokens_override=max_tokens,
                )
            elif selected == "ollama":
                stream = _ollama_stream(
                    prompt,
                    system_prompt,
                    timeout_seconds_override=timeout_seconds,
                    max_tokens_override=max_tokens,
                )
            elif selected == "llama_cpp":
                stream = _llama_cpp_stream(prompt, system_prompt, max_tokens_override=max_tokens)
            else:
                raise RuntimeError("No LLM provider available")

            for token in stream:
                token_count += 1
                text_length += len(token or "")
                yield token
        finally:
            span.set_attribute("gen_ai.response.token_chunk_count", token_count)
            span.set_attribute("gen_ai.response.text_length", text_length)


def is_llm_ready() -> bool:
    return choose_provider() != "none"


def active_provider_name() -> str:
    return choose_provider()
