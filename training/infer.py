"""Sample chat replies from a Tinker base model or a saved sampler checkpoint.

The scheduling loop takes any object with ``complete(messages)``. Tests use a
fake. ``open_sampler`` builds the real Tinker client.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from synthesis.io import append_jsonl, drop_incomplete_tail, load_ids

from training.prompts import InferJob


@dataclass(frozen=True)
class Completion:
    content: str
    stop_reason: str | None = None
    termination: str | None = None


class Sampler(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> Awaitable[Completion]: ...


@dataclass
class InferStats:
    total: int = 0
    skipped: int = 0
    selected: int = 0
    wrote: int = 0
    errors: int = 0
    pending: int = 0


async def run_inference(
    jobs: Sequence[InferJob],
    sampler: Sampler,
    output_dir: Path,
    *,
    concurrency: int = 8,
    batch_size: int = 32,
    limit: int | None = None,
    retry_errors: bool = False,
    overwrite: bool = False,
    model: str | None = None,
    model_path: str | None = None,
    renderer: str | None = None,
) -> InferStats:
    """Write one prediction per job.

    Completed ids in ``predictions.jsonl`` are skipped. Ids in ``errors.jsonl``
    are skipped unless ``retry_errors`` is set. A raised exception from the
    sampler is left unfinished so the same command tries it again.
    """

    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >= 1")
    predictions = output_dir / "predictions.jsonl"
    errors = output_dir / "errors.jsonl"
    failures = output_dir / "failures.jsonl"
    if overwrite:
        for path in (predictions, errors, failures):
            path.unlink(missing_ok=True)
    else:
        for path in (predictions, errors, failures):
            drop_incomplete_tail(path)
    if retry_errors:
        errors.unlink(missing_ok=True)

    done = load_ids(predictions)
    if not retry_errors:
        done |= load_ids(errors)
    pending = [job for job in jobs if job.id not in done]
    selected = pending if limit is None else pending[:limit]
    stats = InferStats(total=len(jobs), skipped=len(jobs) - len(pending), selected=len(selected))
    if not selected:
        _log(f"nothing to do: {stats.skipped} prompts already finished")
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def run_one(job: InferJob) -> None:
        async with semaphore:
            try:
                completion = await sampler.complete(job.messages)
            except Exception as exc:
                outcome: Completion | Exception = exc
            else:
                outcome = completion
        async with write_lock:
            _commit(job, outcome, predictions, errors, failures, stats, model, model_path, renderer)

    for window in _chunks(selected, batch_size):
        tasks = [asyncio.create_task(run_one(job)) for job in window]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise
    return stats


def _commit(
    job: InferJob,
    outcome: Completion | Exception,
    predictions: Path,
    errors: Path,
    failures: Path,
    stats: InferStats,
    model: str | None,
    model_path: str | None,
    renderer: str | None,
) -> None:
    position = stats.wrote + stats.errors + stats.pending + 1
    label = f"[{position}/{stats.selected}]"
    if isinstance(outcome, Completion):
        if not outcome.content.strip():
            append_jsonl(errors, {"id": job.id, "error": "empty completion"})
            stats.errors += 1
            _log(f"{label} error {job.id}: empty completion")
            return
        meta = dict(job.meta)
        meta.update(
            {
                "model": model,
                "model_path": model_path,
                "renderer": renderer,
                "stop_reason": outcome.stop_reason,
                "termination": outcome.termination,
            }
        )
        if job.gold is not None:
            meta["gold"] = job.gold
        messages = [*job.messages, {"role": "assistant", "content": outcome.content}]
        append_jsonl(predictions, {"id": job.id, "messages": messages, "meta": meta})
        stats.wrote += 1
        _log(f"{label} wrote {job.id}")
        return
    append_jsonl(failures, {"id": job.id, "error": f"{type(outcome).__name__}: {outcome}"})
    stats.pending += 1
    _log(f"{label} pending {job.id}: {outcome}")


def _chunks(jobs: Sequence[InferJob], size: int) -> list[list[InferJob]]:
    return [list(jobs[start : start + size]) for start in range(0, len(jobs), size)]


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def require_cookbook() -> None:
    """Raise ``RuntimeError`` with an install hint when Tinker is missing."""

    try:
        import tinker_cookbook  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'Tinker is not installed. Install the training extra: pip install -e ".[train]"'
        ) from exc


def require_api_key() -> None:
    import os

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("Set TINKER_API_KEY before training or inference.")


async def open_sampler(
    *,
    model: str | None,
    renderer_name: str | None,
    model_path: str | None,
    max_tokens: int,
    temperature: float,
    base_url: str | None,
):
    """Create a sampler backed by a Tinker ``SamplingClient``.

    ``model_path`` selects a saved checkpoint. Without it, ``model`` selects
    the base model. The renderer name must be known; pass one or a model name
    whose recommended renderer Tinker can look up.
    """

    require_cookbook()
    require_api_key()
    import tinker
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.renderers import get_renderer, get_text_content

    service = tinker.ServiceClient(base_url=base_url) if base_url else tinker.ServiceClient()
    if model_path:
        client = await service.create_sampling_client_async(model_path=model_path)
    else:
        if not model:
            raise RuntimeError("Pass --model to sample a base model.")
        client = await service.create_sampling_client_async(base_model=model)
    name = renderer_name
    if not name:
        if not model:
            raise RuntimeError("Pass --renderer or --model so the chat template is known.")
        name = get_recommended_renderer_name(model)
    tokenizer = client.get_tokenizer()
    renderer = get_renderer(name, tokenizer, model_name=model)
    params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )

    class _Sampler:
        async def complete(self, messages: list[dict[str, str]]) -> Completion:
            prompt = renderer.build_generation_prompt(messages)
            result = await client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
            sequence = result.sequences[0]
            message, termination = renderer.parse_response(sequence.tokens)
            return Completion(
                content=get_text_content(message),
                stop_reason=_public_name(getattr(sequence, "stop_reason", None)),
                termination=_public_name(termination),
            )

    return _Sampler(), name


def _public_name(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", None)
    if isinstance(raw, str):
        return raw
    return str(value)
