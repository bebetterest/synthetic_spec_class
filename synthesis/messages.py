"""Chat-message checks shared by pipelines and the runner."""

from __future__ import annotations

from typing import Any

from synthesis.types import ParseError

ALLOWED_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})


def normalize_messages(
    messages: Any,
    *,
    require_user: bool = False,
    require_assistant: bool = False,
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` safe to save as training data.

    Each item keeps ``role`` and ``content``. A boolean ``trainable`` field is
    kept when present. Other keys are dropped. Raises ``ParseError`` when the
    list cannot be used as training data. ``developer`` is allowed because the
    Model Spec addresses that role.
    """

    if not isinstance(messages, list) or not messages:
        raise ParseError("messages must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ParseError(f"message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ParseError(f"message {index} has unsupported role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ParseError(f"message {index} content must be a non-empty string")
        item: dict[str, Any] = {"role": role, "content": content}
        if "trainable" in message:
            trainable = message["trainable"]
            if not isinstance(trainable, bool):
                raise ParseError(f"message {index} trainable must be a boolean")
            item["trainable"] = trainable
        normalized.append(item)
    roles = {message["role"] for message in normalized}
    if require_user and "user" not in roles:
        raise ParseError("messages must include a user message")
    if require_assistant and "assistant" not in roles:
        raise ParseError("messages must include an assistant message")
    return normalized


def chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return role and content only, for an API or sampling request.

    ``trainable`` stays in saved training rows. Chat endpoints do not accept it.
    """

    return [{"role": message["role"], "content": message["content"]} for message in messages]
