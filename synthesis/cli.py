"""Command line for synthesis runs.

Run from the repository root:

    python -m synthesis list
    python -m synthesis run <pipeline> --help
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from synthesis.client import ChatClient
from synthesis.io import materialize_train
from synthesis.pipeline import PIPELINES, get_pipeline, load_pipeline_file
from synthesis.runner import _log, run_synthesis

_HELP = """\
usage: python -m synthesis <command>

commands:
  list                      Show available pipelines
  run PIPELINE [options]    Run one pipeline. Use --help after the name for its flags
  materialize OUTPUT_DIR    Rebuild train.jsonl from jobs.jsonl

--pipeline-file FILE may appear anywhere and may be repeated.
"""


def main(argv: list[str] | None = None) -> int:
    import synthesis.pipelines  # noqa: F401  loads @register side effects

    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        pipeline_files, argv = _take_pipeline_files(argv)
        for path in pipeline_files:
            load_pipeline_file(path)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not argv or argv[0] in {"-h", "--help"}:
        print(_HELP, end="")
        return 0
    command, rest = argv[0], argv[1:]
    if command == "list":
        return _list(rest)
    if command == "run":
        return _run(rest)
    if command == "materialize":
        return _materialize(rest)
    print(f"unknown command {command!r}\n", file=sys.stderr)
    print(_HELP, end="", file=sys.stderr)
    return 2


def _take_pipeline_files(argv: list[str]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    rest: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--pipeline-file":
            if index + 1 >= len(argv):
                raise ValueError("--pipeline-file needs a Python file")
            files.append(Path(argv[index + 1]))
            index += 2
            continue
        rest.append(argv[index])
        index += 1
    return files, rest


def _list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m synthesis list", description="Show pipelines.")
    parser.parse_args(argv)
    if not PIPELINES:
        print("no pipelines registered")
        return 0
    width = max(len(name) for name in PIPELINES)
    for name in sorted(PIPELINES):
        print(f"{name:<{width}}  {PIPELINES[name].description}")
    return 0


def _materialize(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m synthesis materialize",
        description="Rebuild train.jsonl from jobs.jsonl.",
    )
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    jobs_path = args.output_dir / "jobs.jsonl"
    if not jobs_path.is_file():
        print(f"no jobs file at {jobs_path}", file=sys.stderr)
        return 2
    count = materialize_train(jobs_path, args.output_dir / "train.jsonl")
    print(f"wrote {count} training rows to {args.output_dir / 'train.jsonl'}")
    return 0


def _run(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m synthesis run PIPELINE [options]\n", file=sys.stderr)
        _list([])
        return 0 if argv else 2
    name, pipeline_argv = argv[0], argv[1:]
    try:
        pipeline = get_pipeline(name)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2
    parser = _run_parser(pipeline)
    args = parser.parse_args(pipeline_argv)
    model = args.model or os.environ.get("SYNTHESIS_MODEL")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not model:
        print("Pass --model or set SYNTHESIS_MODEL.", file=sys.stderr)
        return 2
    if not args.dry_run and not api_key and "api.openai.com" in base_url:
        print("Pass --api-key or set OPENAI_API_KEY.", file=sys.stderr)
        return 2
    client = ChatClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        max_tokens=args.max_tokens,
    )

    async def _execute():
        try:
            return await run_synthesis(
                pipeline,
                args,
                client,
                args.output,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                limit=args.limit,
                retry_errors=args.retry_errors,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        finally:
            await client.aclose()

    try:
        stats = asyncio.run(_execute())
    except KeyboardInterrupt:
        _log("Interrupted. Completed jobs are saved. Run the same command to resume.")
        return 130
    except (OSError, ValueError, KeyError) as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.dry_run:
        return 0
    _log(
        f"done: wrote {stats.wrote}, errors {stats.errors}, pending {stats.pending}, "
        f"skipped {stats.skipped}, train rows {stats.train_rows}"
    )
    _log(f"output: {args.output}")
    return 1 if stats.pending else 0


def _run_parser(pipeline) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m synthesis run {pipeline.name}",
        description=pipeline.description,
    )
    parser.add_argument("--output", required=True, type=Path, help="Directory for JSONL checkpoints.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to SYNTHESIS_MODEL.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="API root such as https://api.openai.com/v1. Defaults to OPENAI_BASE_URL.",
    )
    parser.add_argument("--api-key", default=None, help="Bearer token. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--concurrency", type=int, default=8, help="Maximum in-flight requests (default: 8).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Pending jobs scheduled together (default: 32). Use a value >= concurrency.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many pending jobs.")
    parser.add_argument("--max-tokens", type=int, default=None, help="max_tokens sent on each request.")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds (default: 120).")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries after the first attempt for rate limits and server errors (default: 5).",
    )
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Initial retry delay in seconds (default: 1).")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Run jobs recorded in errors.jsonl again. Successful jobs are still skipped.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete this output directory's checkpoints first.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first pending request and call no API.")
    pipeline.add_arguments(parser)
    return parser
