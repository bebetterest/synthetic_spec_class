"""Command line for Tinker SFT and inference.

    python -m training sft --help
    python -m training infer --help
    python -m training checkpoints LOG_DIR
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from training.checkpoints import last_record, load_run_info
from training.data import SFTSettings, plan_sft
from training.infer import open_sampler, run_inference
from training.prompts import jobs_from_file, jobs_from_prompt
from training.sft import describe, run_sft

_HELP = """\
usage: python -m training <command>

commands:
  sft                       Fine-tune on conversation JSONL
  infer                     Sample from a checkpoint or a base model
  checkpoints LOG_DIR       Print the newest sampler and state paths
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(_HELP, end="")
        return 0
    command, rest = argv[0], argv[1:]
    if command == "sft":
        return _sft(rest)
    if command == "infer":
        return _infer(rest)
    if command == "checkpoints":
        return _checkpoints(rest)
    print(f"unknown command {command!r}\n", file=sys.stderr)
    print(_HELP, end="", file=sys.stderr)
    return 2


def _sft(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training sft",
        description="Supervised fine-tune a Tinker model on conversation JSONL.",
    )
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        type=Path,
        help="train.jsonl files or synthesis directories. Several inputs are concatenated in order.",
    )
    parser.add_argument("--log-dir", required=True, type=Path, help="Tinker log directory. Re-running resumes.")
    parser.add_argument("--model", required=True, help="Tinker base model, for example Qwen/Qwen3-8B.")
    parser.add_argument("--renderer", default=None, help="Chat renderer. Defaults to the model's recommended one.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16, help="Examples per step. Must be <= training rows.")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=0, help="Held-out conversations for NLL eval.")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--train-on-what",
        default="all_assistant_messages",
        help=(
            "Which tokens receive loss. Default trains every assistant message. "
            "customized uses a boolean trainable field on each message."
        ),
    )
    parser.add_argument("--save-every", type=int, default=20, help="Checkpoint cadence in steps. 0 disables.")
    parser.add_argument("--eval-every", type=int, default=0, help="NLL eval cadence. Requires --test-size.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--load-checkpoint", default=None, help="tinker:// weights used when the log dir has no resume state.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--base-url", default=None, help="Override the Tinker service URL.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the log directory before training.")
    parser.add_argument("--dry-run", action="store_true", help="Print the run plan and do not call Tinker.")
    args = parser.parse_args(argv)
    settings = SFTSettings(
        data=tuple(args.data),
        log_dir=args.log_dir,
        model=args.model,
        renderer=args.renderer,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        test_size=args.test_size,
        shuffle_seed=args.shuffle_seed,
        train_on_what=args.train_on_what,
        save_every=args.save_every,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        load_checkpoint=args.load_checkpoint,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        base_url=args.base_url,
        overwrite=args.overwrite,
    )
    try:
        if args.dry_run:
            plan = plan_sft(settings)
            print(describe(plan, settings.renderer))
            return 0
        run_sft(settings)
    except KeyboardInterrupt:
        print("Interrupted. Run the same command to resume from the last checkpoint.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


def _infer(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training infer",
        description="Sample assistant replies with a Tinker renderer.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", default=None, help="One user message.")
    source.add_argument("--input", type=Path, default=None, help="JSONL of messages or prompt rows.")
    parser.add_argument("--output", type=Path, default=None, help="Directory for predictions.jsonl. Required with --input.")
    parser.add_argument("--system", default=None, help="System message for --prompt, and for JSONL rows that use prompt.")
    parser.add_argument("--log-dir", type=Path, default=None, help="SFT log dir. Supplies the newest sampler checkpoint and run info.")
    parser.add_argument("--model-path", default=None, help="tinker:// sampler checkpoint. Overrides --log-dir.")
    parser.add_argument("--model", default=None, help="Base model name. Required when no checkpoint is used.")
    parser.add_argument("--renderer", default=None, help="Chat renderer. Defaults to the SFT run, then the model's recommendation.")
    parser.add_argument("--base", action="store_true", help="Sample the base model instead of a checkpoint.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32, help="Prompts scheduled together. Use a value >= concurrency.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)
    if args.input is not None and args.output is None:
        print("Pass --output when using --input.", file=sys.stderr)
        return 2
    try:
        jobs = (
            jobs_from_file(args.input, args.system)
            if args.input is not None
            else jobs_from_prompt(args.prompt, args.system)
        )
        model, renderer, model_path = _resolve_target(args)
        if args.output is None:
            return asyncio.run(_print_one(jobs[0], model, renderer, model_path, args))
        return asyncio.run(_run_file(jobs, model, renderer, model_path, args))
    except KeyboardInterrupt:
        print("Interrupted. Run the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2


def _resolve_target(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    """Pick the model name, renderer, and sampler checkpoint for one infer command.

    ``--log-dir`` loads the newest sampler checkpoint. ``--base`` samples the
    base model recorded for that run.
    """

    if args.base and args.model_path:
        raise ValueError("Pass either --base or --model-path.")
    if args.log_dir is not None and not args.log_dir.is_dir():
        raise ValueError(f"log dir not found: {args.log_dir}")
    info = load_run_info(args.log_dir) if args.log_dir else None
    model = args.model or (info or {}).get("model")
    renderer = args.renderer or (info or {}).get("renderer")
    if args.base:
        model_path = None
    else:
        model_path = args.model_path
        if model_path is None and args.log_dir is not None:
            row = last_record(args.log_dir, "sampler_path")
            model_path = None if row is None else str(row["sampler_path"])
            if model_path is None:
                raise ValueError(
                    f"No sampler checkpoint in {args.log_dir}. "
                    "Wait for a saved checkpoint, pass --model-path, or pass --base to sample the base model."
                )
    if model_path is None and not model:
        raise ValueError(
            "Pass --model, or --log-dir from an SFT run that recorded synthetic_spec_class_run.json."
        )
    if model_path is not None and not renderer and not model:
        raise ValueError("Pass --renderer or --model so the checkpoint's chat template is known.")
    return model, renderer, model_path


async def _print_one(job, model, renderer, model_path, args) -> int:
    sampler, _resolved = await open_sampler(
        model=model,
        renderer_name=renderer,
        model_path=model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        base_url=args.base_url,
    )
    completion = await sampler.complete(job.messages)
    print(completion.content)
    return 0 if completion.content.strip() else 1


async def _run_file(jobs, model, renderer, model_path, args) -> int:
    sampler, resolved = await open_sampler(
        model=model,
        renderer_name=renderer,
        model_path=model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        base_url=args.base_url,
    )
    stats = await run_inference(
        jobs,
        sampler,
        args.output,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        limit=args.limit,
        retry_errors=args.retry_errors,
        overwrite=args.overwrite,
        model=model,
        model_path=model_path,
        renderer=resolved,
    )
    print(
        f"done: wrote {stats.wrote}, errors {stats.errors}, pending {stats.pending}, skipped {stats.skipped}",
        file=sys.stderr,
    )
    print(f"output: {args.output}", file=sys.stderr)
    return 1 if stats.pending else 0


def _checkpoints(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training checkpoints",
        description="Print the newest sampler and training-state paths in a log directory.",
    )
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args(argv)
    if not args.log_dir.is_dir():
        print(f"log dir not found: {args.log_dir}", file=sys.stderr)
        return 2
    try:
        sampler = last_record(args.log_dir, "sampler_path")
        state = last_record(args.log_dir, "state_path")
        info = load_run_info(args.log_dir)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if info:
        print(f"model: {info.get('model')}")
        print(f"renderer: {info.get('renderer')}")
    print(f"sampler_path: {None if sampler is None else sampler.get('sampler_path')}")
    print(f"state_path: {None if state is None else state.get('state_path')}")
    return 0
