from __future__ import annotations

from typing import Protocol

import httpx


class Generator(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> str: ...


class GenerationError(RuntimeError):
    """Raised when llama-server is unreachable, errors, or returns malformed data."""


class LlamaServerGenerator:
    """Generator backed by llama.cpp's OpenAI-compatible llama-server endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 300.0,
        temperature: float = 0.2,
        model: str = "local",
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.model = model
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Temperature is sent per request on purpose: llm-serve starts the RAG
        # model with --temp 1.0, which is far too hot for grounded citation.
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        url = f"{self.base_url}/v1/chat/completions"
        try:
            response = self._get_client().post(url, json=body)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise GenerationError(f"llama-server generation failed: {exc}") from exc

        return str(content)

    def health(self) -> bool:
        try:
            return self._get_client().get(f"{self.base_url}/health", timeout=5.0).is_success
        except Exception:
            return False

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
