"""Load conversation JSONL and check that a Tinker SFT run can use it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synthesis.io import read_jsonl
from synthesis.messages import normalize_messages
from synthesis.types import ParseError

LR_SCHEDULES = frozenset({"linear", "cosine", "constant"})
TRAIN_ON = frozenset(
    {
        "last_assistant_message",
        "last_assistant_turn",
        "all_assistant_messages",
        "all_messages",
        "all_tokens",
        "all_user_and_system_messages",
        "customized",
    }
)


@dataclass(frozen=True)
class SFTSettings:
    """Knobs for one SFT run. Paths are as the user passed them.

    ``data`` is one path or several. A directory means ``train.jsonl`` inside
    it. Several paths are concatenated, in order, when training starts.
    """

    data: Path | tuple[Path, ...]
    log_dir: Path
    model: str
    renderer: str | None = None
    learning_rate: float = 1e-4
    lr_schedule: str = "linear"
    num_epochs: int = 1
    batch_size: int = 16
    max_length: int = 32768
    lora_rank: int = 32
    test_size: int = 0
    shuffle_seed: int = 0
    train_on_what: str = "all_assistant_messages"
    save_every: int = 20
    eval_every: int = 0
    max_steps: int | None = None
    load_checkpoint: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        data = self.data
        if isinstance(data, Path):
            paths: tuple[Path, ...] = (data,)
        elif isinstance(data, tuple):
            paths = data
        elif isinstance(data, list):
            paths = tuple(data)
        else:
            raise TypeError("data must be a path or a sequence of paths")
        if not paths:
            raise ValueError("Pass --data")
        object.__setattr__(self, "data", paths)


@dataclass(frozen=True)
class SFTPlan:
    """What an SFT command will do, computed without calling Tinker."""

    data: Path
    sources: tuple[Path, ...]
    rows: int
    train_rows: int
    test_rows: int
    renderer: str | None
    settings: SFTSettings


def resolve_data_path(path: Path) -> Path:
    """Return a conversation JSONL path.

    A directory is read as ``<directory>/train.jsonl``, which is what the
    synthesis runner writes.
    """

    if path.is_dir():
        candidate = path / "train.jsonl"
        if not candidate.is_file():
            raise ValueError(f"no train.jsonl in {path}")
        return candidate
    if path.is_file():
        return path
    raise ValueError(f"dataset not found: {path}")


def load_conversations(path: Path, *, train_on_what: str | None = None) -> list[dict]:
    """Return rows that each have a non-empty ``messages`` list.

    Extra keys such as ``id`` and ``meta`` are kept. Message ``trainable``
    flags are kept too. Tinker reads ``messages`` from the file itself; this
    function only checks them.
    """

    rows: list[dict] = []
    for line_number, row in enumerate(read_jsonl(path), start=1):
        try:
            messages = normalize_messages(
                row.get("messages"), require_user=True, require_assistant=True
            )
            if train_on_what is not None:
                _check_trainable(messages, train_on_what)
        except ParseError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        rows.append({**row, "messages": messages})
    if not rows:
        raise ValueError(f"{path} has no conversations")
    return rows


def _check_trainable(messages: list[dict], train_on_what: str) -> None:
    present = ["trainable" in message for message in messages]
    if train_on_what == "customized":
        if not all(present):
            raise ParseError(
                "--train-on-what customized requires a boolean trainable field on every message"
            )
        return
    if any(present):
        raise ParseError(
            "messages set trainable; pass --train-on-what customized, or remove those fields"
        )


def plan_sft(settings: SFTSettings) -> SFTPlan:
    """Validate settings and count rows. Does not contact Tinker."""

    if not settings.model:
        raise ValueError("Pass --model. Example: Qwen/Qwen3-8B")
    if settings.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if settings.lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"--lr-schedule must be one of {', '.join(sorted(LR_SCHEDULES))}")
    if settings.num_epochs < 1:
        raise ValueError("--num-epochs must be >= 1")
    if settings.lora_rank < 1:
        raise ValueError("--lora-rank must be >= 1")
    if settings.max_length < 1:
        raise ValueError("--max-length must be >= 1")
    if settings.save_every < 0 or settings.eval_every < 0:
        raise ValueError("--save-every and --eval-every must be >= 0")
    if settings.max_steps is not None and settings.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if settings.train_on_what not in TRAIN_ON:
        raise ValueError(f"--train-on-what must be one of {', '.join(sorted(TRAIN_ON))}")
    if settings.eval_every > 0 and settings.test_size == 0:
        raise ValueError("Set --test-size when --eval-every is positive, or pass --eval-every 0")

    sources = _resolve_sources(settings.data)
    row_count = 0
    for source in sources:
        row_count += len(load_conversations(source, train_on_what=settings.train_on_what))
    training_file = sources[0] if len(sources) == 1 else settings.log_dir / "combined.jsonl"
    train_rows, test_rows = _split_sizes(row_count, settings.batch_size, settings.test_size)
    return SFTPlan(
        data=training_file,
        sources=sources,
        rows=row_count,
        train_rows=train_rows,
        test_rows=test_rows,
        renderer=settings.renderer,
        settings=settings,
    )


def _resolve_sources(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    sources = tuple(resolve_data_path(path) for path in paths)
    if len(set(sources)) != len(sources):
        raise ValueError("--data lists the same training file more than once")
    return sources


def recorded_data(sources: tuple[Path, ...]) -> str | list[str]:
    """Value stored in ``synthetic_spec_class_run.json`` for the training files."""

    values = [str(path) for path in sources]
    if len(values) == 1:
        return values[0]
    return values


def write_combined_jsonl(sources: tuple[Path, ...], dest: Path) -> None:
    """Concatenate conversation files without rewriting their messages.

    Copying the original lines keeps a ``trainable`` flag that validation
    would otherwise only have checked in memory.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            text += "\n"
        parts.append(text)
    dest.write_text("".join(parts), encoding="utf-8")


def _split_sizes(n_rows: int, batch_size: int, test_size: int) -> tuple[int, int]:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if test_size < 0:
        raise ValueError("--test-size must be >= 0")
    if test_size >= n_rows:
        raise ValueError(f"--test-size {test_size} must be smaller than the dataset ({n_rows} rows)")
    train_rows = n_rows - test_size
    if train_rows < batch_size:
        raise ValueError(
            f"{train_rows} training rows is smaller than --batch-size {batch_size}. "
            "Tinker drops a partial batch, so lower --batch-size."
        )
    return train_rows, test_size
