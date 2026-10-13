"""Schedule pipeline jobs with concurrency, batches, and resume."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from synthesis.client import build_chat_body
from synthesis.io import append_jsonl, drop_incomplete_tail, load_ids, materialize_train, train_row
from synthesis.messages import normalize_messages
from synthesis.pipeline import Pipeline
from synthesis.types import ChatRequest, Example, FatalAPIError, Job, ParseError, Produced, RetryableError


class _Completer(Protocol):
    model: str | None

    def complete(self, request: ChatRequest) -> Awaitable[str]: ...


@dataclass
class RunStats:
    total: int = 0
    skipped: int = 0
    selected: int = 0
    wrote: int = 0
    errors: int = 0
    pending: int = 0
    train_rows: int = 0


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path

    @property
    def jobs(self) -> Path:
        return self.output_dir / "jobs.jsonl"

    @property
    def train(self) -> Path:
        return self.output_dir / "train.jsonl"

    @property
    def errors(self) -> Path:
        return self.output_dir / "errors.jsonl"

    @property
    def failures(self) -> Path:
        return self.output_dir / "failures.jsonl"


async def run_synthesis(
    pipeline: Pipeline,
    args: argparse.Namespace,
    client: _Completer,
    output_dir: Path,
    *,
    concurrency: int = 8,
    batch_size: int = 32,
    limit: int | None = None,
    retry_errors: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> RunStats:
    """Run ``pipeline`` into ``output_dir``.

    Jobs already present in ``jobs.jsonl`` are skipped. Jobs present in
    ``errors.jsonl`` are skipped too, unless ``retry_errors`` is set. A
    retryable API failure is written to ``failures.jsonl`` and left unfinished,
    so the next run tries it again.

    ``batch_size`` is how many pending jobs are scheduled at once.
    ``concurrency`` is how many ``complete`` calls may be in flight, including
    several calls from one multi-step job. Set ``batch_size`` greater than or
    equal to ``concurrency``.
    """

    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >= 1")
    jobs = _load_jobs(pipeline, args)
    paths = RunPaths(output_dir)
    if overwrite and not dry_run:
        for path in (paths.jobs, paths.train, paths.errors, paths.failures):
            path.unlink(missing_ok=True)
    if not dry_run:
        for path in (paths.jobs, paths.errors, paths.failures):
            drop_incomplete_tail(path)
    if retry_errors and not dry_run:
        # Drop recorded parse and fatal errors so those jobs run again.
        # A crash before they are retried leaves them unfinished, and the next
        # run picks them up without this flag.
        paths.errors.unlink(missing_ok=True)

    done = load_ids(paths.jobs)
    if not retry_errors:
        done |= load_ids(paths.errors)
    pending = [job for job in jobs if job.id not in done]
    selected = pending if limit is None else pending[:limit]
    stats = RunStats(
        total=len(jobs),
        skipped=len(jobs) - len(pending),
        selected=len(selected),
    )
    if dry_run:
        _print_dry_run(pipeline, args, client, stats, selected)
        return stats

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    stats.train_rows = materialize_train(paths.jobs, paths.train)
    if not selected:
        _log(f"nothing to do: {stats.skipped} jobs already finished, train rows {stats.train_rows}")
        return stats

    semaphore = asyncio.Semaphore(concurrency)
    limited = _LimitedCompleter(client, semaphore)
    wrote_lock = asyncio.Lock()

    async def run_one(job: Job) -> None:
        outcome = await _generate(pipeline, job, args, limited)
        async with wrote_lock:
            _commit(paths, pipeline, job, outcome, client, stats)

    try:
        for window in _chunks(selected, batch_size):
            tasks = [asyncio.create_task(run_one(job)) for job in window]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                raise
    finally:
        stats.train_rows = materialize_train(paths.jobs, paths.train)
    return stats


def _load_jobs(pipeline: Pipeline, args: argparse.Namespace) -> list[Job]:
    jobs = list(pipeline.jobs(args))
    seen: set[str] = set()
    for job in jobs:
        if job.id in seen:
            raise ValueError(f"duplicate job id {job.id!r} from pipeline {pipeline.name!r}")
        seen.add(job.id)
    return jobs


async def _generate(
    pipeline: Pipeline,
    job: Job,
    args: argparse.Namespace,
    client: _Completer,
) -> tuple[Any, list[dict[str, Any]]] | Exception:
    try:
        produced = _as_produced(await pipeline.produce(job, args, client))
        raw_text = produced.raw if isinstance(produced.raw, str) else ""
        return produced.raw, _checked_examples(job, produced.examples, raw_text)
    except (ParseError, FatalAPIError, RetryableError) as exc:
        return exc
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return ParseError(f"{type(exc).__name__}: {exc}")


def _as_produced(value: Any) -> Produced:
    if isinstance(value, Produced):
        return value
    if isinstance(value, list):
        return Produced(examples=value)
    raise ParseError("produce() must return examples or Produced")


def _checked_examples(job: Job, examples: Iterable[Example], raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, Example):
            raise ParseError(f"parse() item {index} is not an Example", raw=raw)
        try:
            messages = normalize_messages(
                example.messages, require_user=True, require_assistant=True
            )
        except ParseError as exc:
            if exc.raw is None:
                exc.raw = raw
            raise
        rows.append(
            {
                "id": f"{job.id}#{index}",
                "messages": messages,
                "meta": dict(example.meta),
            }
        )
    if not rows:
        raise ParseError("parse() returned no examples", raw=raw)
    return rows


def _commit(
    paths: RunPaths,
    pipeline: Pipeline,
    job: Job,
    outcome: tuple[str, list[dict[str, Any]]] | Exception,
    client: _Completer,
    stats: RunStats,
) -> None:
    position = stats.wrote + stats.errors + stats.pending + 1
    label = f"[{position}/{stats.selected}]"
    if isinstance(outcome, tuple):
        raw, examples = outcome
        model = getattr(client, "model", None)
        for example in examples:
            meta = example["meta"]
            meta.setdefault("pipeline", pipeline.name)
            if model:
                meta.setdefault("model", model)
        append_jsonl(
            paths.jobs,
            {"id": job.id, "pipeline": pipeline.name, "examples": examples, "raw": raw},
        )
        for example in examples:
            append_jsonl(paths.train, train_row(job.id, example))
        stats.wrote += 1
        _log(f"{label} wrote {job.id} examples={len(examples)}")
        return
    if isinstance(outcome, RetryableError):
        append_jsonl(
            paths.failures,
            {"id": job.id, "pipeline": pipeline.name, "error": str(outcome)},
        )
        stats.pending += 1
        _log(f"{label} pending {job.id}: {outcome}")
        return
    raw = outcome.raw if isinstance(outcome, ParseError) else None
    append_jsonl(
        paths.errors,
        {"id": job.id, "pipeline": pipeline.name, "error": str(outcome), "raw": raw},
    )
    stats.errors += 1
    _log(f"{label} error {job.id}: {outcome}")


def _print_dry_run(
    pipeline: Pipeline,
    args: argparse.Namespace,
    client: _Completer,
    stats: RunStats,
    selected: list[Job],
) -> None:
    _log(
        f"dry run: pipeline={pipeline.name} pending={stats.selected} "
        f"skipped={stats.skipped} total={stats.total}"
    )
    if not selected:
        return
    job = selected[0]
    try:
        request = pipeline.preview(job, args)
    except NotImplementedError:
        request = None
    _log(f"next job: {job.id}")
    if request is None:
        _log("dry-run stops before produce(); this pipeline builds its requests there")
        return
    body = build_chat_body(
        request,
        model=getattr(client, "model", None),
        max_tokens=getattr(client, "max_tokens", None),
    )
    _log("request body:")
    sys.stderr.write(json.dumps(body, ensure_ascii=False, indent=2) + "\n")


class _LimitedCompleter:
    """Bound every ``complete`` call to the run-wide concurrency limit."""

    def __init__(self, inner: _Completer, semaphore: asyncio.Semaphore) -> None:
        self._inner = inner
        self._semaphore = semaphore

    async def complete(self, request: ChatRequest) -> str:
        async with self._semaphore:
            return await self._inner.complete(request)


def _chunks(items: list[Job], size: int) -> Iterable[list[Job]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _log(message: str) -> None:
    print(message, file=sys.stderr)
