import asyncio
import json
from pathlib import Path

import pytest

from argparse import Namespace

from training.checkpoints import last_sampler_path, write_run_info
from training.cli import _resolve_target, main
from training.data import SFTSettings, plan_sft, write_combined_jsonl
from training.sft import resume_conflict
from training.infer import Completion, run_inference
from training.prompts import jobs_from_file, split_gold
from training.sft import describe


def _conversation(user: str, assistant: str, row_id: str = "row") -> dict:
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_plan_reads_a_synthesis_directory_and_rejects_a_short_batch(tmp_path: Path):
    data = tmp_path / "run"
    data.mkdir()
    _write_jsonl(data / "train.jsonl", [_conversation("Hi", "Hello", f"id-{i}") for i in range(4)])
    plan = plan_sft(
        SFTSettings(data=data, log_dir=tmp_path / "logs", model="Qwen/Qwen3-8B", batch_size=2)
    )
    assert plan.rows == 4
    assert plan.train_rows == 4
    assert plan.data.name == "train.jsonl"
    with pytest.raises(ValueError, match="batch-size"):
        plan_sft(SFTSettings(data=data, log_dir=tmp_path / "logs", model="Qwen/Qwen3-8B", batch_size=8))


def test_plan_combines_files_and_checks_trainable(tmp_path: Path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_jsonl(first, [_conversation("One", "A", "a1"), _conversation("Two", "B", "a2")])
    _write_jsonl(
        second,
        [
            {
                "id": "b1",
                "messages": [
                    {"role": "user", "content": "Three", "trainable": False},
                    {"role": "assistant", "content": "C", "trainable": True},
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="customized"):
        plan_sft(
            SFTSettings(
                data=(first, second),
                log_dir=tmp_path / "logs",
                model="Qwen/Qwen3-8B",
                batch_size=2,
            )
        )
    plain = tmp_path / "plain"
    plain.mkdir()
    _write_jsonl(plain / "train.jsonl", [_conversation("One", "A", "a1"), _conversation("Two", "B", "a2")])
    plan = plan_sft(
        SFTSettings(data=(first, plain), log_dir=tmp_path / "logs", model="Qwen/Qwen3-8B", batch_size=2)
    )
    assert plan.rows == 4
    assert plan.train_rows == 4
    assert plan.data.name == "combined.jsonl"
    assert not plan.data.exists()
    write_combined_jsonl(plan.sources, plan.data)
    combined = [json.loads(line) for line in plan.data.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in combined] == ["a1", "a2", "a1", "a2"]

    flagged = tmp_path / "flagged.jsonl"
    _write_jsonl(
        flagged,
        [
            {
                "id": "c1",
                "messages": [
                    {"role": "user", "content": "Three", "trainable": False},
                    {"role": "assistant", "content": "C", "trainable": True},
                ],
            }
        ],
    )
    custom = plan_sft(
        SFTSettings(
            data=flagged,
            log_dir=tmp_path / "logs",
            model="Qwen/Qwen3-8B",
            batch_size=1,
            train_on_what="customized",
        )
    )
    assert custom.rows == 1
    with pytest.raises(ValueError, match="trainable"):
        plan_sft(
            SFTSettings(
                data=first,
                log_dir=tmp_path / "logs",
                model="Qwen/Qwen3-8B",
                batch_size=1,
                train_on_what="customized",
            )
        )


def test_resume_conflict_requires_the_same_run(tmp_path: Path):
    log_dir = tmp_path / "logs"
    write_run_info(log_dir, {"model": "Qwen/Qwen3-8B", "renderer": "qwen3", "data": "/tmp/train.jsonl"})
    assert (
        resume_conflict(
            log_dir,
            model="Qwen/Qwen3-8B",
            renderer="qwen3",
            data="/tmp/train.jsonl",
            overwrite=False,
        )
        is None
    )
    message = resume_conflict(
        log_dir,
        model="Qwen/Qwen3-32B",
        renderer="qwen3",
        data="/tmp/train.jsonl",
        overwrite=False,
    )
    assert message is not None
    assert "overwrite" in message
    assert (
        resume_conflict(
            log_dir,
            model="Qwen/Qwen3-32B",
            renderer="qwen3",
            data="/tmp/train.jsonl",
            overwrite=True,
        )
        is None
    )


def test_plan_requires_a_held_out_split_for_eval(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_conversation("Hi", "Hello", f"id-{i}") for i in range(4)])
    with pytest.raises(ValueError, match="test-size"):
        plan_sft(
            SFTSettings(
                data=path,
                log_dir=tmp_path / "logs",
                model="Qwen/Qwen3-8B",
                batch_size=2,
                eval_every=10,
            )
        )


def test_split_gold_keeps_earlier_turns():
    prefix, gold = split_gold(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "Two"},
            {"role": "assistant", "content": "B"},
        ]
    )
    assert gold == "B"
    assert [message["content"] for message in prefix] == ["Be brief.", "One", "A", "Two"]


def test_jobs_from_train_file_store_the_reference(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_conversation("Question", "Reference", "ex-1")])
    jobs = jobs_from_file(path, system=None)
    assert jobs[0].id == "ex-1"
    assert jobs[0].gold == "Reference"
    assert jobs[0].messages == [{"role": "user", "content": "Question"}]


def test_last_sampler_path_is_the_newest_row(tmp_path: Path):
    rows = [
        {"name": "a", "sampler_path": "tinker://run/sampler_weights/a", "state_path": "tinker://run/weights/a"},
        {"name": "final", "sampler_path": "tinker://run/sampler_weights/final", "state_path": "tinker://run/weights/final"},
    ]
    (tmp_path / "checkpoints.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert last_sampler_path(tmp_path) == "tinker://run/sampler_weights/final"


class FakeSampler:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.inflight = 0
        self.max_inflight = 0

    async def complete(self, messages):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.calls.append(messages[-1]["content"])
        await asyncio.sleep(0.02)
        self.inflight -= 1
        if messages[-1]["content"] == "bad":
            return Completion(content="  ")
        return Completion(content=f"answer {messages[-1]['content']}", stop_reason="stop")


def test_inference_resume_concurrency_and_gold(tmp_path: Path):
    jobs = jobs_from_file(
        _jsonl(
            tmp_path / "in.jsonl",
            [
                _conversation("keep", "gold-keep", "keep"),
                _conversation("bad", "gold-bad", "bad"),
                _conversation("later", "gold-later", "later"),
            ],
        ),
        system=None,
    )
    sampler = FakeSampler()
    output = tmp_path / "infer"

    async def scenario() -> None:
        first = await run_inference(jobs, sampler, output, concurrency=2, batch_size=2)
        assert first.wrote == 2
        assert first.errors == 1
        assert sampler.max_inflight == 2
        second = await run_inference(jobs, sampler, output, concurrency=2, batch_size=2)
        assert second.selected == 0
        assert sampler.calls.count("keep") == 1

    asyncio.run(scenario())
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    kept = next(row for row in rows if row["id"] == "keep")
    assert kept["messages"][-1]["content"] == "answer keep"
    assert kept["meta"]["gold"] == "gold-keep"
    assert kept["messages"][0]["role"] == "user"


def test_sft_dry_run_and_checkpoints_command(tmp_path: Path, capsys):
    data = tmp_path / "train.jsonl"
    _write_jsonl(data, [_conversation("Hi", "Hello", f"id-{i}") for i in range(3)])
    log_dir = tmp_path / "logs"
    code = main(
        [
            "sft",
            "--data",
            str(data),
            "--log-dir",
            str(log_dir),
            "--model",
            "Qwen/Qwen3-8B",
            "--batch-size",
            "2",
            "--dry-run",
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "Qwen/Qwen3-8B" in text
    assert "train=3" in text
    assert "does not fill a batch" in text
    assert not log_dir.exists()

    log_dir.mkdir()
    write_run_info(log_dir, {"model": "Qwen/Qwen3-8B", "renderer": "qwen3"})
    (log_dir / "checkpoints.jsonl").write_text(
        json.dumps({"name": "final", "sampler_path": "tinker://run/sampler_weights/final", "state_path": "tinker://run/weights/final"})
        + "\n",
        encoding="utf-8",
    )
    assert main(["checkpoints", str(log_dir)]) == 0
    shown = capsys.readouterr().out
    assert "tinker://run/sampler_weights/final" in shown
    assert "renderer: qwen3" in shown
    plan = plan_sft(SFTSettings(data=data, log_dir=log_dir, model="Qwen/Qwen3-8B", batch_size=2, renderer="qwen3"))
    assert "renderer: qwen3" in describe(plan, "qwen3")


def test_resolve_target_requires_a_sampler_checkpoint(tmp_path: Path):
    log_dir = tmp_path / "logs"
    write_run_info(log_dir, {"model": "Qwen/Qwen3-8B", "renderer": "qwen3"})
    args = Namespace(base=False, model_path=None, log_dir=log_dir, model=None, renderer=None)
    with pytest.raises(ValueError, match="sampler checkpoint"):
        _resolve_target(args)


def test_resolve_target_base_uses_the_recorded_model(tmp_path: Path):
    log_dir = tmp_path / "logs"
    write_run_info(log_dir, {"model": "Qwen/Qwen3-8B", "renderer": "qwen3"})
    (log_dir / "checkpoints.jsonl").write_text(
        json.dumps({"sampler_path": "tinker://run/sampler_weights/final"}) + "\n",
        encoding="utf-8",
    )
    base = Namespace(base=True, model_path=None, log_dir=log_dir, model=None, renderer=None)
    model, renderer, model_path = _resolve_target(base)
    assert (model, renderer, model_path) == ("Qwen/Qwen3-8B", "qwen3", None)

    checkpoint = Namespace(base=False, model_path=None, log_dir=log_dir, model=None, renderer=None)
    _, _, model_path = _resolve_target(checkpoint)
    assert model_path == "tinker://run/sampler_weights/final"

    both = Namespace(base=True, model_path="tinker://x", log_dir=None, model="Qwen/Qwen3-8B", renderer=None)
    with pytest.raises(ValueError, match="--base"):
        _resolve_target(both)


def _jsonl(path: Path, rows: list[dict]) -> Path:
    _write_jsonl(path, rows)
    return path
