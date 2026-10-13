"""Pipeline contract.

A synthesis chain is a :class:`Pipeline`. The runner schedules jobs. The
pipeline decides how many model calls each job makes.

The usual chain implements :meth:`Pipeline.jobs`, :meth:`Pipeline.request`,
and :meth:`Pipeline.parse`. That is one model call per job. A multi-step chain
overrides :meth:`Pipeline.produce` and calls the client once per step.

Register a class with :func:`register`. A module in ``synthesis/pipelines/``
is imported on startup. A file outside the package is loaded with
``--pipeline-file``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Iterable
from pathlib import Path

from synthesis.types import ChatRequest, Example, Job, ParseError, Produced

PIPELINES: dict[str, type[Pipeline]] = {}


class Pipeline:
    """One synthesis chain.

    Reserved runner flags that a pipeline must not redefine: ``--output``,
    ``--model``, ``--base-url``, ``--api-key``, ``--concurrency``,
    ``--batch-size``, ``--limit``, ``--max-tokens``, ``--timeout``,
    ``--max-retries``, ``--retry-backoff``, ``--retry-errors``,
    ``--overwrite``, ``--dry-run``, ``--pipeline-file``.
    """

    name: str = ""
    description: str = ""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add flags that apply only to this pipeline."""

    def jobs(self, args: argparse.Namespace) -> Iterable[Job]:
        """Yield jobs in a stable order. Job ids are resume keys."""

        raise NotImplementedError

    def request(self, job: Job, args: argparse.Namespace) -> ChatRequest:
        """Build the chat request for a one-call pipeline."""

        raise NotImplementedError

    def parse(self, job: Job, content: str, args: argparse.Namespace) -> list[Example]:
        """Turn one assistant message into training examples.

        Raise ``ParseError`` when the response cannot be used. The runner
        records that error and skips the job on later runs.
        """

        raise NotImplementedError

    async def produce(self, job: Job, args: argparse.Namespace, client: object) -> Produced | list[Example]:
        """Run one job and return its examples.

        The default is one ``request`` and one ``parse``. Override this to call
        ``client.complete`` several times. Calls inside one job run one after
        another unless the override gathers them. ``--concurrency`` limits
        ``complete`` calls across the whole run, including gathered ones.

        The job is checkpointed only after this method returns. A failure on a
        later step runs the job again from the first call.
        """

        request = self.request(job, args)
        if not isinstance(request, ChatRequest):
            raise ParseError("request() must return a ChatRequest")
        complete = getattr(client, "complete", None)
        if complete is None:
            raise ParseError("client has no complete()")
        content = await complete(request)
        if not isinstance(content, str):
            raise ParseError("complete() must return a string")
        try:
            examples = list(self.parse(job, content, args))
        except ParseError as exc:
            if exc.raw is None:
                exc.raw = content
            raise
        except Exception as exc:
            raise ParseError(f"{type(exc).__name__}: {exc}", raw=content) from exc
        return Produced(examples=examples, raw=content)

    def preview(self, job: Job, args: argparse.Namespace) -> ChatRequest | None:
        """Return the request a dry-run should print.

        The default is :meth:`request`. A multi-step pipeline can return its
        first request, or ``None`` when there is nothing to print.
        """

        return self.request(job, args)


def register(cls: type[Pipeline]) -> type[Pipeline]:
    """Register a pipeline class under ``cls.name``."""

    name = cls.name
    if not isinstance(name, str) or not name.isidentifier() or name != name.lower():
        raise ValueError(
            f"{cls.__name__}.name must be a lower-case identifier, got {name!r}"
        )
    if not cls.description:
        raise ValueError(f"{cls.__name__}.description is empty")
    existing = PIPELINES.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"pipeline {name!r} is already registered by {existing.__name__}")
    PIPELINES[name] = cls
    return cls


def get_pipeline(name: str) -> Pipeline:
    try:
        cls = PIPELINES[name]
    except KeyError:
        known = ", ".join(sorted(PIPELINES)) or "(none)"
        raise KeyError(f"unknown pipeline {name!r}; known pipelines: {known}") from None
    return cls()


def load_pipeline_file(path: Path) -> None:
    """Import a Python file so its ``@register`` classes become available."""

    if not path.is_file():
        raise ValueError(f"pipeline file not found: {path}")
    resolved = path.resolve()
    module_name = "synthesis_user_" + "".join(ch if ch.isalnum() else "_" for ch in path.stem)
    module_name = f"{module_name}_{abs(hash(resolved)) & 0xffffffff:x}"
    existing = sys.modules.get(module_name)
    existing_file = getattr(existing, "__file__", None)
    if existing is not None and existing_file and Path(existing_file).resolve() == resolved:
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import pipeline file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
