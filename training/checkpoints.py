"""Read the checkpoint log Tinker writes, without importing Tinker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synthesis.io import read_jsonl

CHECKPOINTS_NAME = "checkpoints.jsonl"
RUN_INFO_NAME = "synthetic_spec_class_run.json"


def load_checkpoints(log_dir: Path) -> list[dict[str, Any]]:
    path = log_dir / CHECKPOINTS_NAME
    if not path.is_file():
        return []
    return list(read_jsonl(path))


def last_record(log_dir: Path, key: str) -> dict[str, Any] | None:
    """Return the last checkpoint row that has a non-empty ``key``.

    ``checkpoints.jsonl`` is append-only, so the last match is the newest.
    Use ``sampler_path`` for inference and ``state_path`` for a training resume.
    """

    matches = [row for row in load_checkpoints(log_dir) if row.get(key)]
    if not matches:
        return None
    return matches[-1]


def last_sampler_path(log_dir: Path) -> str | None:
    row = last_record(log_dir, "sampler_path")
    if row is None:
        return None
    return str(row["sampler_path"])


def load_run_info(log_dir: Path) -> dict[str, Any] | None:
    path = log_dir / RUN_INFO_NAME
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def write_run_info(log_dir: Path, info: dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / RUN_INFO_NAME
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
