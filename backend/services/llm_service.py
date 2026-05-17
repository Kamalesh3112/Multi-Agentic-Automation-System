from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

import httpx

from backend.config import get_settings


LOGGER = logging.getLogger("backend.services.llm")
SETTINGS = get_settings()


class LLMProvider(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str: ...

    async def stream(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]: ...

    async def health_check(self) -> bool: ...


def _pick_env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value if value is not None else default


def _structured_log(event: str, **fields: object) -> None:
    LOGGER.info(event, extra={"event": event, **fields})


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    timeout_seconds: float
    max_retries: int
    base_retry_delay_seconds: float
    openrouter_api_key: str | None
    openrouter_base_url: str
    ollama_base_url: str
    fallback_providers: tuple[str, ...]
    openrouter_gpt_fallback_model: str
    openrouter_claude_fallback_model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        fallback_raw = _pick_env("LLM_FALLBACK_PROVIDERS", "openrouter")
        fallback = tuple(p.strip().lower() for p in fallback_raw.split(",") if p.strip())
        return cls(
            provider=_pick_env("LLM_PROVIDER", "ollama").lower(),
            model=_pick_env("LLM_MODEL", "llama3.1:8b"),
            timeout_seconds=float(_pick_env("LLM_TIMEOUT_SECONDS", "45")),
            max_retries=int(_pick_env("LLM_MAX_RETRIES", "3")),
            base_retry_delay_seconds=float(_pick_env("LLM_RETRY_BASE_DELAY_SECONDS", "1")),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_base_url=_pick_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            ollama_base_url=_pick_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            fallback_providers=fallback,
            openrouter_gpt_fallback_model=_pick_env("OPENROUTER_GPT_FALLBACK_MODEL", "openai/gpt-4o-mini"),
            openrouter_claude_fallback_model=_pick_env("OPENROUTER_CLAUDE_FALLBACK_MODEL", "anthropic/claude-3.5-sonnet"),
        )


class OpenRouterProvider:
    def __init__(self, client: httpx.AsyncClient, config: LLMConfig) -> None:
        self._client = client
        self._config = config

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        if not self._config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._config.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if SETTINGS.app_name:
            headers["X-Title"] = SETTINGS.app_name

        response = await self._client.post(
            f"{self._config.openrouter_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        body = response.json()
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Unexpected OpenRouter response schema") from exc

    async def stream(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        if not self._config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        headers = {
            "Authorization": f"Bearer {self._config.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if SETTINGS.app_name:
            headers["X-Title"] = SETTINGS.app_name

        async with self._client.stream(
            "POST",
            f"{self._config.openrouter_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    delta = data["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue

    async def health_check(self) -> bool:
        if not self._config.openrouter_api_key:
            return False
        try:
            response = await self._client.get(
                f"{self._config.openrouter_base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {self._config.openrouter_api_key}"},
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False


class OllamaProvider:
    def __init__(self, client: httpx.AsyncClient, config: LLMConfig) -> None:
        self._client = client
        self._config = config

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = await self._client.post(
            f"{self._config.ollama_base_url.rstrip('/')}/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
        output = body.get("response")
        if not isinstance(output, str):
            raise ValueError("Unexpected Ollama response schema")
        return output

    async def stream(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with self._client.stream(
            "POST",
            f"{self._config.ollama_base_url.rstrip('/')}/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response")
                if isinstance(token, str) and token:
                    yield token
                if chunk.get("done") is True:
                    break

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(f"{self._config.ollama_base_url.rstrip('/')}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False


class LLMService:
    def __init__(self, config: LLMConfig | None = None, client: httpx.AsyncClient | None = None) -> None:
        self._config = config or LLMConfig.from_env()
        timeout = httpx.Timeout(self._config.timeout_seconds)
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._providers: dict[str, LLMProvider] = {
            "openrouter": OpenRouterProvider(self._client, self._config),
            "ollama": OllamaProvider(self._client, self._config),
        }
        self._model_aliases = {
            "phi-3": _pick_env("OLLAMA_MODEL_PHI3", "phi3:mini"),
            "qwen2.5-coder": _pick_env("OLLAMA_MODEL_QWEN25_CODER", "qwen2.5-coder:7b"),
            "llama3": _pick_env("OLLAMA_MODEL_LLAMA3", "llama3.1:8b"),
            "openrouter-gpt-fallback": self._config.openrouter_gpt_fallback_model,
            "openrouter-claude-fallback": self._config.openrouter_claude_fallback_model,
        }

    def _build_provider(self, provider_name: str) -> LLMProvider:
        provider = self._providers.get(provider_name.lower())
        if provider is None:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        return provider

    def select_provider(self, requested_provider: str | None = None) -> str:
        return (requested_provider or self._config.provider).lower()

    def _resolve_model(self, model: str | None) -> str:
        candidate = model or self._config.model
        return self._model_aliases.get(candidate.lower(), candidate)

    def _provider_order(self, selected_provider: str) -> tuple[str, ...]:
        order: list[str] = [selected_provider]
        for provider in self._config.fallback_providers:
            if provider != selected_provider:
                order.append(provider)
        return tuple(order)

    def _fallback_models_for_provider(self, provider: str, selected_model: str) -> tuple[str, ...]:
        if provider != "openrouter":
            return (selected_model,)
        return (
            selected_model,
            self._config.openrouter_gpt_fallback_model,
            self._config.openrouter_claude_fallback_model,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def provider_health_check(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for name, provider in self._providers.items():
            result[name] = await provider.health_check()
        _structured_log("llm_provider_health", **result)
        return result

    async def generate_response(
        self,
        prompt: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        selected_provider = self.select_provider(provider)
        selected_model = self._resolve_model(model)

        last_error: Exception | None = None
        for candidate_provider in self._provider_order(selected_provider):
            active_provider = self._build_provider(candidate_provider)
            for candidate_model in self._fallback_models_for_provider(candidate_provider, selected_model):
                for attempt in range(1, self._config.max_retries + 1):
                    try:
                        _structured_log(
                            "llm_generate_start",
                            provider=candidate_provider,
                            model=candidate_model,
                            attempt=attempt,
                        )
                        text = await active_provider.generate(
                            prompt,
                            model=candidate_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            system_prompt=system_prompt,
                        )
                        _structured_log(
                            "llm_generate_success",
                            provider=candidate_provider,
                            model=candidate_model,
                            attempt=attempt,
                            output_chars=len(text),
                        )
                        return text
                    except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as exc:
                        last_error = exc
                        retryable = attempt < self._config.max_retries
                        LOGGER.warning(
                            "llm_generate_attempt_failed",
                            extra={
                                "event": "llm_generate_attempt_failed",
                                "provider": candidate_provider,
                                "model": candidate_model,
                                "attempt": attempt,
                                "retryable": retryable,
                                "error_type": exc.__class__.__name__,
                            },
                        )
                        if not retryable:
                            break
                        delay = self._config.base_retry_delay_seconds * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)

        LOGGER.exception(
            "llm_generate_failed",
            extra={
                "event": "llm_generate_failed",
                "provider": selected_provider,
                "model": selected_model,
                "max_retries": self._config.max_retries,
            },
        )
        raise RuntimeError("LLM generation failed after retries") from last_error

    async def stream_response(
        self,
        prompt: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        selected_provider = self.select_provider(provider)
        selected_model = self._resolve_model(model)

        last_error: Exception | None = None
        for candidate_provider in self._provider_order(selected_provider):
            active_provider = self._build_provider(candidate_provider)
            for candidate_model in self._fallback_models_for_provider(candidate_provider, selected_model):
                try:
                    _structured_log(
                        "llm_stream_start",
                        provider=candidate_provider,
                        model=candidate_model,
                    )
                    async for chunk in active_provider.stream(
                        prompt,
                        model=candidate_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                    ):
                        yield chunk
                    _structured_log(
                        "llm_stream_success",
                        provider=candidate_provider,
                        model=candidate_model,
                    )
                    return
                except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as exc:
                    last_error = exc
                    LOGGER.warning(
                        "llm_stream_failed",
                        extra={
                            "event": "llm_stream_failed",
                            "provider": candidate_provider,
                            "model": candidate_model,
                            "error_type": exc.__class__.__name__,
                        },
                    )
                    continue
        raise RuntimeError("LLM stream failed for all providers/models") from last_error

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        return await self.generate_response(
            prompt,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )


_SERVICE: LLMService | None = None


def get_llm_service() -> LLMService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = LLMService()
    return _SERVICE


async def generate(
    prompt: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    system_prompt: str | None = None,
) -> str:
    service = get_llm_service()
    return await service.generate(
        prompt,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


async def generate_response(
    prompt: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    system_prompt: str | None = None,
) -> str:
    service = get_llm_service()
    return await service.generate_response(
        prompt,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


async def stream_response(
    prompt: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    service = get_llm_service()
    async for chunk in service.stream_response(
        prompt,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    ):
        yield chunk


def select_provider(requested_provider: str | None = None) -> str:
    service = get_llm_service()
    return service.select_provider(requested_provider)


async def provider_health_check() -> dict[str, bool]:
    service = get_llm_service()
    return await service.provider_health_check()
