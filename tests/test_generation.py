from __future__ import annotations

import json

import httpx
import pytest

from physics_rag.generation import GenerationError, LlamaServerGenerator


def _generator(handler, **kwargs) -> LlamaServerGenerator:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LlamaServerGenerator(base_url="http://llama.test", client=client, **kwargs)


def test_generate_returns_content_and_sends_expected_body() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    generator = _generator(handler, temperature=0.2)

    result = generator.generate("What is entropy?", system="Cite sources", max_tokens=512)

    assert result == "hello"
    assert captured["url"] == "http://llama.test/v1/chat/completions"

    body = captured["body"]
    assert isinstance(body, dict)
    # The server is launched with --temp 1.0; the client's low temperature must win.
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 512
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "Cite sources"},
        {"role": "user", "content": "What is entropy?"},
    ]


def test_generate_omits_system_message_when_none() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    _generator(handler).generate("q")

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["messages"] == [{"role": "user", "content": "q"}]


def test_generate_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    with pytest.raises(GenerationError):
        _generator(handler).generate("test")


def test_generate_raises_on_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"foo": "bar"})

    with pytest.raises(GenerationError):
        _generator(handler).generate("test")


def test_health_true_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200)

    assert _generator(handler).health() is True


def test_health_false_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed")

    assert _generator(handler).health() is False


def test_generate_disables_thinking_by_default() -> None:
    """Reasoning models spend the token budget on reasoning_content, not content."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert _generator(handler).generate("q") == "ok"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_generate_keeps_thinking_when_explicitly_enabled() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _generator(handler, enable_thinking=True).generate("q")
    body = captured["body"]
    assert isinstance(body, dict)
    assert "chat_template_kwargs" not in body


def test_generate_raises_when_answer_was_truncated() -> None:
    """A truncated answer reads as a bad citation; it must fail loudly instead."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "The Fried"}, "finish_reason": "length"}]},
        )

    with pytest.raises(GenerationError, match="truncated"):
        _generator(handler).generate("q", max_tokens=1024)
