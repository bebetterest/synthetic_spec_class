"""Turn conversation rows into inference prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthesis.io import read_jsonl
from synthesis.messages import chat_messages, normalize_messages
from synthesis.types import ParseError


@dataclass(frozen=True)
class InferJob:
    """One sampling request.

    ``messages`` is the prompt prefix. ``gold`` is the assistant message that
    was removed from the end of a training row, when there was one.
    """

    id: str
    messages: list[dict[str, str]]
    gold: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def jobs_from_prompt(prompt: str, system: str | None) -> list[InferJob]:
    if not prompt.strip():
        raise ValueError("--prompt is empty")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    prefix, gold = split_gold(messages)
    return [InferJob(id="prompt", messages=prefix, gold=gold)]


def jobs_from_file(path: Path, system: str | None) -> list[InferJob]:
    """Load a JSONL file of ``messages`` or ``prompt`` rows.

    A trailing assistant message is removed and kept as ``gold``. That lets a
    ``train.jsonl`` file be used as an inference input without training the
    model to continue after its own answer. ``--system`` is prepended only to
    rows that use ``prompt`` rather than ``messages``.
    """

    if not path.is_file():
        raise ValueError(f"input file not found: {path}")
    jobs: list[InferJob] = []
    seen: set[str] = set()
    for line_number, row in enumerate(read_jsonl(path), start=1):
        job_id = row.get("id")
        if not isinstance(job_id, str) or not job_id:
            job_id = f"row-{line_number}"
        if job_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate id {job_id!r}")
        seen.add(job_id)
        messages = _row_messages(row, system, path, line_number)
        try:
            prefix, gold = split_gold(messages)
        except ParseError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        meta = row.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError(f"{path}:{line_number}: meta must be an object")
        jobs.append(InferJob(id=job_id, messages=prefix, gold=gold, meta=dict(meta)))
    if not jobs:
        raise ValueError(f"{path} has no prompts")
    return jobs


def split_gold(messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str | None]:
    """Drop a trailing assistant message and return it as the reference answer."""

    normalized = normalize_messages(messages, require_user=False, require_assistant=False)
    gold = None
    if normalized[-1]["role"] == "assistant":
        gold = normalized[-1]["content"]
        normalized = normalized[:-1]
    if not normalized or not any(message["role"] == "user" for message in normalized):
        raise ParseError("prompt must include a user message before the assistant reply")
    return chat_messages(normalized), gold


def _row_messages(row: dict[str, Any], system: str | None, path: Path, line_number: int) -> list[dict[str, Any]]:
    has_messages = "messages" in row
    has_prompt = "prompt" in row
    if has_messages == has_prompt:
        raise ValueError(f"{path}:{line_number}: provide exactly one of messages or prompt")
    if has_prompt:
        prompt = row["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{line_number}: prompt must be a non-empty string")
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages
    messages = row["messages"]
    if not isinstance(messages, list):
        raise ValueError(f"{path}:{line_number}: messages must be a list")
    return list(messages)
