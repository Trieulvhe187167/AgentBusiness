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
from typing import Generator

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_llama_model = None


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


def _llama_cpp_stream(prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
    model = _load_llama_cpp()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    stream = model.create_chat_completion(
        messages=messages,
        max_tokens=settings.llm_max_tokens,
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


def _ollama_stream(prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
    base = settings.effective_ollama_base_url.rstrip("/")
    if not base:
        raise RuntimeError("RAG_OLLAMA_BASE_URL is not configured")

    url = f"{base}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "temperature": settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "num_predict": settings.llm_max_tokens,
        },
    }

    timeout = httpx.Timeout(settings.ollama_timeout_seconds)
    with httpx.stream("POST", url, json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token


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


def _openai_stream(prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
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
        "model": settings.openai_model,
        "input": input_items,
        "temperature": settings.llm_temperature,
        "max_output_tokens": settings.llm_max_tokens,
        "stream": False,
    }

    timeout = httpx.Timeout(settings.openai_timeout_seconds)
    response = httpx.post(url, headers=headers, json=body, timeout=timeout)
    response.raise_for_status()
    text = _extract_openai_text(response.json())
    for chunk in _safe_chunk_text(text):
        yield chunk


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    return text.strip()


def _gemini_stream(prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
    if not settings.gemini_api_key:
        raise RuntimeError("RAG_GEMINI_API_KEY is not configured")

    base = settings.gemini_base_url.rstrip("/")
    model = settings.gemini_model
    url = f"{base}/models/{model}:generateContent?key={settings.gemini_api_key}"

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": settings.llm_temperature,
            "topP": settings.llm_top_p,
            "maxOutputTokens": settings.llm_max_tokens,
        },
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    timeout = httpx.Timeout(settings.gemini_timeout_seconds)
    response = httpx.post(url, json=body, timeout=timeout)
    response.raise_for_status()
    text = _extract_gemini_text(response.json())
    for chunk in _safe_chunk_text(text):
        yield chunk


def _openai_compatible_stream(prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
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
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "stream": True,
    }

    timeout = httpx.Timeout(settings.llm_timeout_seconds)
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


def generate_stream(prompt: str, system_prompt: str = "", provider: str | None = None) -> Generator[str, None, None]:
    selected = (provider or choose_provider()).lower().strip()

    if selected == "openai":
        yield from _openai_stream(prompt, system_prompt)
        return
    if selected == "openai_compatible":
        yield from _openai_compatible_stream(prompt, system_prompt)
        return
    if selected == "gemini":
        yield from _gemini_stream(prompt, system_prompt)
        return
    if selected == "ollama":
        yield from _ollama_stream(prompt, system_prompt)
        return
    if selected == "llama_cpp":
        yield from _llama_cpp_stream(prompt, system_prompt)
        return

    raise RuntimeError("No LLM provider available")


def is_llm_ready() -> bool:
    return choose_provider() != "none"


def active_provider_name() -> str:
    return choose_provider()
