"""Records passed between a pipeline and the runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Job:
    """One API call.

    ``id`` is the resume key. It must be stable across runs and must not
    contain newlines. ``input`` is private to the pipeline.
    """

    id: str
    input: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or any(c in self.id for c in "\r\n"):
            raise ValueError(f"job id must be a non-empty single-line string, got {self.id!r}")


@dataclass
class ChatRequest:
    """One chat-completions request.

    ``extra`` is merged into the JSON body after the standard fields, so a
    pipeline can pass vendor-specific keys such as ``response_format``.
    """

    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Example:
    """One trainable conversation produced from a job.

    Messages are ``role`` and ``content``. A boolean ``trainable`` key is kept
    so a later SFT run can use ``--train-on-what customized``.
    """

    messages: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Produced:
    """What :meth:`synthesis.pipeline.Pipeline.produce` returns.

    ``examples`` become training rows. ``raw`` is stored on the job checkpoint
    for debugging. A multi-step pipeline can set it to a list of step outputs.
    Returning a plain list of examples is also accepted.
    """

    examples: list[Example]
    raw: Any = None


class ParseError(Exception):
    """The model responded, but the pipeline could not turn it into examples.

    Recorded in ``errors.jsonl`` and skipped on later runs until
    ``--retry-errors``.
    """

    def __init__(self, message: str, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


class FatalAPIError(Exception):
    """The request failed and retrying it unchanged will not help.

    Recorded in ``errors.jsonl`` and skipped on later runs until
    ``--retry-errors``. Typical causes are HTTP 4xx (except 408, 409, and 429)
    and an empty completion.
    """


class RetryableError(Exception):
    """The request failed after the configured retries.

    Not marked done. The same command picks the job up again.
    """
