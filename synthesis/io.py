"""JSONL checkpoint files.

``jobs.jsonl`` is the source of truth for resume. ``train.jsonl`` is rebuilt
from it and is the file to fine-tune on. Both are append-only during a run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def drop_incomplete_tail(path: Path) -> None:
    """Drop a trailing partial line left when a process dies mid-append.

    A finished record always ends with a newline. Resume ignores an unfinished
    tail so that job is run again instead of failing the whole checkpoint.
    """

    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    head, _newline, _tail = data.rpartition(b"\n")
    path.write_bytes(head + (b"\n" if head else b""))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def load_ids(path: Path, key: str = "id") -> set[str]:
    return {str(record[key]) for record in read_jsonl(path) if record.get(key)}


def train_row(job_id: str, example: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example["id"],
        "job_id": job_id,
        "messages": example["messages"],
        "meta": example.get("meta") or {},
    }


def materialize_train(jobs_path: Path, train_path: Path) -> int:
    """Rewrite ``train.jsonl`` so it matches ``jobs.jsonl`` exactly."""

    train_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = train_path.with_suffix(train_path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for job in read_jsonl(jobs_path):
            job_id = str(job["id"])
            for example in job.get("examples") or []:
                handle.write(json.dumps(train_row(job_id, example), ensure_ascii=False) + "\n")
                count += 1
    temporary.replace(train_path)
    return count
