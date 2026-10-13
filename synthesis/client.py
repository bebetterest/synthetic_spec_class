"""OpenAI-compatible chat completions client."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from synthesis.types import ChatRequest, FatalAPIError, RetryableError

# 408/409/429 are client status codes that are still worth retrying.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def build_chat_body(
    request: ChatRequest,
    *,
    model: str | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Build the JSON body for ``/chat/completions``.

    Omitted optional fields stay out of the body. Reasoning endpoints often
    reject an explicit temperature, so the default is to leave it unset.
    Keys in ``request.extra`` are applied last.
    """

    body: dict[str, Any] = {"messages": request.messages}
    chosen_model = request.model or model
    if chosen_model:
        body["model"] = chosen_model
    if request.temperature is not None:
        body["temperature"] = request.temperature
    chosen_max_tokens = request.max_tokens if request.max_tokens is not None else max_tokens
    if chosen_max_tokens is not None:
        body["max_tokens"] = chosen_max_tokens
    body.update(request.extra)
    return body


class ChatClient:
    """Async client for one OpenAI-compatible ``/chat/completions`` endpoint.

    ``base_url`` may be either ``https://host/v1`` or the full completions URL.
    An empty API key omits the Authorization header for local servers.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str | None,
        timeout: float = 120,
        max_retries: int = 5,
        retry_backoff: float = 1.0,
        max_tokens: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        trust_env: bool = True,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._url = _completions_url(base_url)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            transport=transport,
            trust_env=trust_env,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, request: ChatRequest) -> str:
        body = build_chat_body(request, model=self.model, max_tokens=self.max_tokens)
        if "model" not in body:
            raise FatalAPIError("no model set; pass --model or set SYNTHESIS_MODEL")
        last_error: RetryableError | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                return await self._once(body)
            except RetryableError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                await asyncio.sleep(self.retry_backoff * (2**attempt))
        assert last_error is not None
        raise last_error

    async def _once(self, body: dict[str, Any]) -> str:
        try:
            response = await self._client.post(self._url, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(f"request failed: {exc}") from exc
        if response.status_code in _RETRYABLE_STATUS:
            raise RetryableError(
                f"HTTP {response.status_code}: {_snippet(response.text)}"
            )
        if response.status_code >= 400:
            raise FatalAPIError(f"HTTP {response.status_code}: {_snippet(response.text)}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RetryableError(f"response is not JSON: {exc}") from exc
        return _message_text(payload)


def _completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("base_url is empty")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _message_text(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise FatalAPIError(f"response has no message content: {exc}") from exc
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise FatalAPIError("model returned an empty completion")
    return content


def _snippet(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."
