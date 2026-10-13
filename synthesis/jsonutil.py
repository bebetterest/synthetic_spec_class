"""Tolerant JSON object parsing for model responses."""

from __future__ import annotations

import json
from typing import Any

from synthesis.types import ParseError


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response.

    Accepts a raw object or an object wrapped in a ``` fence. If the response
    has prose around the object, the outermost ``{...}`` span is parsed.
    """

    candidate = _strip_fence(text.strip())
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise ParseError("response is not a JSON object", raw=text) from None
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ParseError(f"response is not a JSON object: {exc}", raw=text) from None
    if not isinstance(value, dict):
        raise ParseError("response JSON must be an object", raw=text)
    return value


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
