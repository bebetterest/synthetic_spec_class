"""Default template: complete each JSONL prefix and keep the reply as SFT.

Copy this file and edit ``read``, ``prompt``, and ``parse``.

    python -m synthesis run --template path/to/copy.py --output runs/name --input rows.jsonl
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from synthesis.io import read_jsonl
from synthesis.messages import chat_messages, normalize_messages
from synthesis.types import ChatRequest, Example, ParseError

name = "prompts"
description = "Complete each JSONL row and append the assistant reply."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path, help="JSONL file of chat prefixes.")
    parser.add_argument(
        "--system",
        default=None,
        help="System message prepended to every request and saved in the training example.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Omit to leave it unset. A row may override this.",
    )


def read(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """Yield one record per JSONL row."""

    path = Path(args.input)
    if not path.is_file():
        raise ValueError(f"input file not found: {path}")
    for line_number, row in enumerate(read_jsonl(path), start=1):
        item = _row_input(row, args, line_number)
        item["id"] = _row_id(row, line_number, path)
        yield item


def prompt(item: dict[str, Any], args: argparse.Namespace) -> ChatRequest:
    """Send the prefix. ``trainable`` stays on the saved row and is left off the request."""

    del args
    return ChatRequest(
        messages=chat_messages(item["messages"]),
        temperature=item["temperature"],
        model=item["model"],
    )


def parse(item: dict[str, Any], content: str, args: argparse.Namespace) -> list[Example]:
    """Append the reply and keep the row's meta."""

    del args
    text = content.strip()
    if not text:
        raise ParseError("empty completion", raw=content)
    messages = [*item["messages"], {"role": "assistant", "content": text}]
    normalized = normalize_messages(messages, require_user=True, require_assistant=True)
    return [Example(messages=normalized, meta=dict(item["meta"]))]


def _row_id(row: dict[str, Any], line_number: int, path: Path) -> str:
    job_id = row.get("id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError(f"{path}:{line_number}: each row needs a non-empty string id")
    return job_id


def _row_input(row: dict[str, Any], args: argparse.Namespace, line_number: int) -> dict[str, Any]:
    messages = _prefix_messages(row, line_number)
    if args.system:
        messages = [{"role": "system", "content": args.system}, *messages]
    temperature = row.get("temperature", args.temperature)
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise ValueError(f"line {line_number}: temperature must be a number")
    model = row.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError(f"line {line_number}: model must be a string")
    meta = row.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError(f"line {line_number}: meta must be an object")
    return {
        "messages": normalize_messages(messages, require_user=True, require_assistant=False),
        "temperature": None if temperature is None else float(temperature),
        "model": model,
        "meta": dict(meta),
    }


def _prefix_messages(row: dict[str, Any], line_number: int) -> list[dict[str, Any]]:
    has_messages = "messages" in row
    has_prompt = "prompt" in row
    if has_messages == has_prompt:
        raise ValueError(f"line {line_number}: provide exactly one of messages or prompt")
    if has_prompt:
        prompt = row["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"line {line_number}: prompt must be a non-empty string")
        return [{"role": "user", "content": prompt}]
    messages = row["messages"]
    if not isinstance(messages, list):
        raise ValueError(f"line {line_number}: messages must be a list")
    return list(messages)
