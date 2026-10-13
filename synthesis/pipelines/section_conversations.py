"""Synthesize conversations that demonstrate one Markdown policy section."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from synthesis.jsonutil import parse_json_object
from synthesis.markdown_sections import split_sections
from synthesis.messages import normalize_messages
from synthesis.pipeline import Pipeline, register
from synthesis.types import ChatRequest, Example, Job, ParseError

# Tokens use <<NAME>> so a section body that contains braces or $ cannot break
# the prompt. Edit this string to change the synthesis behavior.
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


@register
class SectionConversationsPipeline(Pipeline):
    """One API call per Markdown section, several conversations per call.

    Use this on ``model_spec/model_spec.md`` or the constitution. Heading level
    2 is the default because both documents use level 1 for parts and level 2
    for the principles inside them.
    """

    name = "section_conversations"
    description = "From each Markdown section, synthesize conversations that follow it."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
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

    def jobs(self, args: argparse.Namespace) -> Iterable[Job]:
        _validate(args)
        source = Path(args.source)
        text = source.read_text(encoding="utf-8")
        for section in split_sections(text, args.heading_level):
            body = section.body
            if len(body) > args.max_section_chars:
                body = body[: args.max_section_chars].rstrip() + "\n\n[truncated]"
            yield Job(
                id=f"{source.name}/{section.id}",
                input={
                    "source": str(source),
                    "section_id": section.id,
                    "title": section.title,
                    "level": section.level,
                    "ancestors": list(section.ancestors),
                    "body": body,
                },
            )

    def request(self, job: Job, args: argparse.Namespace) -> ChatRequest:
        ancestors = job.input["ancestors"]
        prompt = (
            PROMPT.replace("<<TITLE>>", job.input["title"])
            .replace("<<ANCESTORS>>", " > ".join(ancestors) if ancestors else "(none)")
            .replace("<<SOURCE>>", job.input["source"])
            .replace("<<N>>", str(args.examples_per_section))
            .replace("<<BODY>>", job.input["body"])
        )
        extra = {} if args.no_json_mode else {"response_format": {"type": "json_object"}}
        return ChatRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            extra=extra,
        )

    def parse(self, job: Job, content: str, args: argparse.Namespace) -> list[Example]:
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
            "source": job.input["source"],
            "section_id": job.input["section_id"],
            "section_title": job.input["title"],
            "heading_level": job.input["level"],
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
