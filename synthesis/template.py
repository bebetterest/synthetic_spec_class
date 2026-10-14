"""Turn a template module into a pipeline.

A template is a Python file with three functions:

* ``read(args)`` yields one record per model call. A record is a dict with a
  string ``id``, or a :class:`~synthesis.types.Job`.
* ``prompt(item, args)`` turns that record into the request. Return a
  :class:`~synthesis.types.ChatRequest`, a user-message string, or a list of
  chat messages. A plain list keeps ``role`` and ``content`` only, so a
  ``trainable`` flag stays out of the API call.
* ``parse(item, content, args)`` turns the reply into SFT conversations.
  Return a list of :class:`~synthesis.types.Example`, a list of message
  lists, or one message list.

Optional ``add_arguments(parser)`` adds flags for that template. ``name`` and
``description`` are the CLI name and the line shown by ``list``. Copy a file
in ``synthesis/templates/`` and edit the three functions.
"""

from __future__ import annotations

import argparse
from typing import Any

from synthesis.messages import chat_messages
from synthesis.pipeline import PIPELINES, Pipeline, load_pipeline_file, register
from synthesis.types import ChatRequest, Example, Job, ParseError

_FUNCTIONS = ("read", "prompt", "parse")


def is_template(module: object) -> bool:
    return all(callable(getattr(module, name, None)) for name in _FUNCTIONS)


def register_template(module: object, *, default_name: str | None = None, replace: bool = False) -> str:
    """Register a template module and return its CLI name."""

    if not is_template(module):
        missing = [name for name in _FUNCTIONS if not callable(getattr(module, name, None))]
        raise ValueError(
            f"{getattr(module, '__file__', module)} needs {', '.join(missing)}()"
        )
    name = getattr(module, "name", None) or default_name
    if not isinstance(name, str) or not name.isidentifier() or name != name.lower():
        raise ValueError(
            f"template name must be a lower-case identifier, got {name!r}. "
            "Set name = \"...\" in the file."
        )
    description = getattr(module, "description", None)
    if not isinstance(description, str) or not description.strip():
        doc = (getattr(module, "__doc__", None) or "").strip().splitlines()
        description = doc[0].strip() if doc else f"Template {name}."
    existing = PIPELINES.get(name)
    if existing is not None and getattr(existing, "module", None) is module:
        return name
    if existing is not None and replace:
        del PIPELINES[name]

    module_ref = module

    class Chain(Pipeline):
        module = module_ref

        def add_arguments(self, parser: argparse.ArgumentParser) -> None:
            add = getattr(self.module, "add_arguments", None)
            if add is not None:
                add(parser)

        def jobs(self, args: argparse.Namespace):
            for index, item in enumerate(self.module.read(args)):
                yield _job_from_read(item, index)

        def request(self, job: Job, args: argparse.Namespace) -> ChatRequest:
            return _request_from_prompt(self.module.prompt(job.input, args))

        def parse(self, job: Job, content: str, args: argparse.Namespace) -> list[Example]:
            try:
                value = self.module.parse(job.input, content, args)
            except ParseError:
                raise
            except Exception as exc:
                raise ParseError(f"{type(exc).__name__}: {exc}", raw=content) from exc
            return _examples_from_parse(value, content)

    Chain.name = name
    Chain.description = description.strip()
    Chain.__name__ = f"Template_{name}"
    Chain.__qualname__ = Chain.__name__
    register(Chain)
    return name


def load_template_file(path) -> str:
    """Import a template file, register it, and return its CLI name."""

    from pathlib import Path

    file_path = Path(path)
    module = load_pipeline_file(file_path)
    if not is_template(module):
        raise ValueError(f"{file_path} needs read(), prompt(), and parse()")
    return register_template(module, default_name=file_path.stem, replace=True)


def _job_from_read(item: Any, index: int) -> Job:
    if isinstance(item, Job):
        data = dict(item.input)
        data.setdefault("id", item.id)
        return Job(id=item.id, input=data)
    if not isinstance(item, dict):
        raise ValueError(f"read() item {index} must be a dict or a Job, got {type(item).__name__}")
    job_id = item.get("id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError(f"read() item {index} needs a non-empty string id")
    return Job(id=job_id, input=dict(item))


def _request_from_prompt(value: Any) -> ChatRequest:
    if isinstance(value, ChatRequest):
        return value
    if isinstance(value, str):
        if not value.strip():
            raise ParseError("prompt() returned an empty string")
        return ChatRequest(messages=[{"role": "user", "content": value}])
    if isinstance(value, list):
        try:
            messages = chat_messages(value)
        except (KeyError, TypeError) as exc:
            raise ParseError("prompt() messages must include role and content") from exc
        return ChatRequest(messages=messages)
    raise ParseError(f"prompt() must return a ChatRequest, a string, or messages, got {type(value).__name__}")


def _examples_from_parse(value: Any, raw: str) -> list[Example]:
    if isinstance(value, Example):
        return [value]
    if not isinstance(value, list) or not value:
        raise ParseError("parse() must return at least one conversation", raw=raw)
    if all(_is_message(item) for item in value):
        return [Example(messages=value)]
    examples: list[Example] = []
    for index, item in enumerate(value):
        if isinstance(item, Example):
            examples.append(item)
        elif isinstance(item, list):
            examples.append(Example(messages=item))
        elif isinstance(item, dict) and isinstance(item.get("messages"), list):
            meta = item.get("meta") or {}
            if not isinstance(meta, dict):
                raise ParseError(f"parse() item {index} meta must be an object", raw=raw)
            examples.append(Example(messages=item["messages"], meta=dict(meta)))
        else:
            raise ParseError(
                f"parse() item {index} must be an Example, a message list, or a messages object",
                raw=raw,
            )
    return examples


def _is_message(value: Any) -> bool:
    return isinstance(value, dict) and "role" in value and "content" in value
