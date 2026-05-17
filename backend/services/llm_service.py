from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

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

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=_pick_env("LLM_PROVIDER", "ollama").lower(),
            model=_pick_env("LLM_MODEL", "llama3.1:8b"),
            timeout_seconds=float(_pick_env("LLM_TIMEOUT_SECONDS", "45")),
            max_retries=int(_pick_env("LLM_MAX_RETRIES", "3")),
            base_retry_delay_seconds=float(_pick_env("LLM_RETRY_BASE_DELAY_SECONDS", "1")),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_base_url=_pick_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            ollama_base_url=_pick_env("OLLAMA_BASE_URL", "http://localhost:11434"),
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


class LLMService:
    def __init__(self, config: LLMConfig | None = None, client: httpx.AsyncClient | None = None) -> None:
        self._config = config or LLMConfig.from_env()
        timeout = httpx.Timeout(self._config.timeout_seconds)
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._provider = self._build_provider(self._config.provider)

    def _build_provider(self, provider_name: str) -> LLMProvider:
        if provider_name == "openrouter":
            return OpenRouterProvider(self._client, self._config)
        if provider_name == "ollama":
            return OllamaProvider(self._client, self._config)
        raise ValueError(f"Unsupported LLM provider: {provider_name}")

    async def close(self) -> None:
        await self._client.aclose()

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
        selected_provider = (provider or self._config.provider).lower()
        selected_model = model or self._config.model
        active_provider = self._provider if selected_provider == self._config.provider else self._build_provider(selected_provider)

        last_error: Exception | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                _structured_log(
                    "llm_generate_start",
                    provider=selected_provider,
                    model=selected_model,
                    attempt=attempt,
                )
                text = await active_provider.generate(
                    prompt,
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
                _structured_log(
                    "llm_generate_success",
                    provider=selected_provider,
                    model=selected_model,
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
                        "provider": selected_provider,
                        "model": selected_model,
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
