import json

import httpx
import pytest

from synthesis.client import ChatClient, build_chat_body
from synthesis.types import ChatRequest, FatalAPIError, RetryableError


def test_build_chat_body_omits_unset_fields_and_applies_extra_last():
    body = build_chat_body(
        ChatRequest(
            messages=[{"role": "user", "content": "Hi"}],
            extra={"response_format": {"type": "json_object"}},
        ),
        model="demo",
        max_tokens=None,
    )
    assert body == {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "demo",
        "response_format": {"type": "json_object"},
    }
    assert "temperature" not in body


def test_completions_url_accepts_a_root_or_a_full_path():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    async def run(base_url: str) -> str:
        client = ChatClient(
            base_url=base_url,
            api_key="test",
            model="demo",
            max_retries=0,
            transport=httpx.MockTransport(handler),
            trust_env=False,
        )
        try:
            return await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await client.aclose()

    assert pytest.importorskip("asyncio").run(run("https://example.com/v1")) == "ok"
    assert pytest.importorskip("asyncio").run(run("https://example.com/v1/chat/completions")) == "ok"
    assert calls["n"] == 2


def test_retry_then_success_and_fatal_status():
    state = {"n": 0}

    def retry_handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 3:
            return httpx.Response(429, text="slow down")
        body = json.loads(request.content.decode())
        assert body["model"] == "demo"
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    def fatal_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, text="nope")

    async def scenario() -> None:
        client = ChatClient(
            base_url="https://example.com/v1",
            api_key=None,
            model="demo",
            max_retries=2,
            retry_backoff=0,
            transport=httpx.MockTransport(retry_handler),
            trust_env=False,
        )
        try:
            text = await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await client.aclose()
        assert text == "done"
        assert state["n"] == 3

        fatal = ChatClient(
            base_url="https://example.com/v1",
            api_key=None,
            model="demo",
            max_retries=5,
            retry_backoff=0,
            transport=httpx.MockTransport(fatal_handler),
            trust_env=False,
        )
        try:
            with pytest.raises(FatalAPIError):
                await fatal.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await fatal.aclose()

        exhausted = ChatClient(
            base_url="https://example.com/v1",
            api_key=None,
            model="demo",
            max_retries=1,
            retry_backoff=0,
            transport=httpx.MockTransport(lambda request: httpx.Response(503, text="down")),
            trust_env=False,
        )
        try:
            with pytest.raises(RetryableError):
                await exhausted.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await exhausted.aclose()

    import asyncio

    asyncio.run(scenario())


def test_list_content_parts_are_joined():
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": [{"type": "text", "text": "a"}, "b"]}}]},
        )

    async def scenario() -> str:
        client = ChatClient(
            base_url="https://example.com/v1",
            api_key=None,
            model="demo",
            max_retries=0,
            transport=httpx.MockTransport(handler),
            trust_env=False,
        )
        try:
            return await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await client.aclose()

    import asyncio

    assert asyncio.run(scenario()) == "ab"
