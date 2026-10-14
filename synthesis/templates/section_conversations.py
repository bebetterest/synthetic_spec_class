"""Default template: one Markdown section becomes several SFT conversations.

Copy this file and edit ``read``, ``prompt``, and ``parse``. ``read`` yields
one record per model call, ``prompt`` builds the request, and ``parse`` turns
the reply into training conversations.

    python -m synthesis run --template path/to/copy.py --output runs/name --source policy.md
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from synthesis.jsonutil import parse_json_object
from synthesis.markdown_sections import split_sections
from synthesis.messages import normalize_messages
from synthesis.types import ChatRequest, Example, ParseError

name = "section_conversations"
description = "From each Markdown section, synthesize conversations that follow it."

# Tokens use <<NAME>> so a section body that contains braces or $ cannot break
# the prompt. Edit this string to change what the model is asked to write.
PROMPT = """You write supervised-fine-tuning conversations that demonstrate a model behavior policy.

Policy section
Title: <<TITLE>>
Ancestors: <<ANCESTORS>>
Source: <<SOURCE>>

<section>
<<BODY>>
</section>

Write <<N>> diverse conversations that a general-purpose assistant should handle according to this section.

Rules for the conversations:
- Each conversation is a realistic request and the assistant reply that follows this section. Add a system or developer message only when the section's behavior depends on one.
- Show the behavior. Do not quote, name, or explain this policy document inside the conversation.
- Vary the setting across the set: direct requests, ambiguous requests, requests that should be refused or only partly answered, and ordinary requests that should be answered in full when this section allows that.
- The assistant reply is the text the model should learn to produce. The reply itself must follow the section.
- Do not include analysis, rubrics, or labels outside the JSON object.

Return a JSON object with this shape and no other top-level keys:
{"conversations": [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]}
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, type=Path, help="Markdown spec or constitution.")
    parser.add_argument("--heading-level", type=int, default=2, help="Heading level to split on (default: 2).")
    parser.add_argument(
        "--examples-per-section",
        type=int,
        default=4,
        help="Conversations to request from the model for each section (default: 4).",
    )
    parser.add_argument(
        "--max-section-chars",
        type=int,
        default=12000,
        help="Truncate section text sent to the model after this many characters (default: 12000).",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Do not send response_format=json_object. Use this for servers that reject it.",
    )


def read(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """Yield one record per section. The record is what prompt and parse receive."""

    _validate(args)
    source = Path(args.source)
    text = source.read_text(encoding="utf-8")
    for section in split_sections(text, args.heading_level):
        body = section.body
        if len(body) > args.max_section_chars:
            body = body[: args.max_section_chars].rstrip() + "\n\n[truncated]"
        yield {
            "id": f"{source.name}/{section.id}",
            "source": str(source),
            "section_id": section.id,
            "title": section.title,
            "level": section.level,
            "ancestors": list(section.ancestors),
            "body": body,
        }


def prompt(item: dict[str, Any], args: argparse.Namespace) -> ChatRequest:
    """Fill the section into the request sent to the model."""

    ancestors = item["ancestors"]
    text = (
        PROMPT.replace("<<TITLE>>", item["title"])
        .replace("<<ANCESTORS>>", " > ".join(ancestors) if ancestors else "(none)")
        .replace("<<SOURCE>>", item["source"])
        .replace("<<N>>", str(args.examples_per_section))
        .replace("<<BODY>>", item["body"])
    )
    extra = {} if args.no_json_mode else {"response_format": {"type": "json_object"}}
    return ChatRequest(
        messages=[{"role": "user", "content": text}],
        temperature=args.temperature,
        extra=extra,
    )


def parse(item: dict[str, Any], content: str, args: argparse.Namespace) -> list[Example]:
    """Turn the JSON reply into one SFT conversation per item."""

    data = parse_json_object(content)
    conversations = data.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ParseError("JSON object has no conversations", raw=content)
    limit = max(args.examples_per_section * 2, 1)
    if len(conversations) > limit:
        raise ParseError(
            f"expected at most {limit} conversations, got {len(conversations)}",
            raw=content,
        )
    meta = {
        "source": item["source"],
        "section_id": item["section_id"],
        "section_title": item["title"],
        "heading_level": item["level"],
    }
    examples: list[Example] = []
    for index, conversation in enumerate(conversations):
        if not isinstance(conversation, dict):
            raise ParseError(f"conversation {index} is not an object", raw=content)
        messages = normalize_messages(
            conversation.get("messages"),
            require_user=True,
            require_assistant=True,
        )
        examples.append(Example(messages=messages, meta=dict(meta)))
    return examples


def _validate(args: argparse.Namespace) -> None:
    source = Path(args.source)
    if not source.is_file():
        raise ValueError(f"source file not found: {source}")
    if args.heading_level < 1 or args.heading_level > 6:
        raise ValueError("--heading-level must be from 1 to 6")
    if args.examples_per_section < 1:
        raise ValueError("--examples-per-section must be >= 1")
    if args.max_section_chars < 1:
        raise ValueError("--max-section-chars must be >= 1")
