import argparse
import asyncio
import json
from pathlib import Path

import pytest

from synthesis.runner import run_synthesis
from synthesis.types import ChatRequest, Example, Job, ParseError, Produced, RetryableError
from synthesis.pipeline import Pipeline


class EchoPipeline(Pipeline):
    name = "echo"
    description = "Test pipeline."

    def jobs(self, args: argparse.Namespace):
        for index in range(args.n):
            yield Job(id=f"job-{index}", input={})

    def request(self, job: Job, args: argparse.Namespace) -> ChatRequest:
        del args
        return ChatRequest(messages=[{"role": "user", "content": job.id}])

    def parse(self, job: Job, content: str, args: argparse.Namespace) -> list[Example]:
        del args
        if content.startswith("BAD"):
            raise ParseError("unusable", raw=content)
        return [
            Example(
                messages=[
                    {"role": "user", "content": job.id},
                    {"role": "assistant", "content": content},
                ]
            )
        ]


class FakeClient:
    def __init__(self, replies: dict[str, str | Exception] | None = None) -> None:
        self.model = "fake-model"
        self.max_tokens = None
        self.replies = replies or {}
        self.calls: list[str] = []
        self.inflight = 0
        self.max_inflight = 0

    async def complete(self, request: ChatRequest) -> str:
        job_id = request.messages[-1]["content"]
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.calls.append(job_id)
        await asyncio.sleep(0.02)
        self.inflight -= 1
        reply = self.replies.get(job_id, f"answer {job_id}")
        if isinstance(reply, Exception):
            raise reply
        return reply


def _args(n: int) -> argparse.Namespace:
    return argparse.Namespace(n=n)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_batch_concurrency_resume_and_errors(tmp_path: Path):
    client = FakeClient(
        {
            "job-1": ParseError("unusable"),
            "job-2": RetryableError("HTTP 503: down"),
        }
    )
    output = tmp_path / "run"

    async def first() -> None:
        stats = await run_synthesis(
            EchoPipeline(),
            _args(4),
            client,
            output,
            concurrency=2,
            batch_size=2,
        )
        assert stats.wrote == 2
        assert stats.errors == 1
        assert stats.pending == 1
        assert stats.selected == 4

    asyncio.run(first())
    assert client.max_inflight == 2
    assert len(client.calls) == 4
    jobs = _read(output / "jobs.jsonl")
    assert {row["id"] for row in jobs} == {"job-0", "job-3"}
    train = _read(output / "train.jsonl")
    assert [row["id"] for row in train] == ["job-0#0", "job-3#0"]
    assert train[0]["messages"][1]["content"] == "answer job-0"
    assert train[0]["meta"]["model"] == "fake-model"
    assert _read(output / "errors.jsonl")[0]["id"] == "job-1"
    assert _read(output / "failures.jsonl")[0]["id"] == "job-2"

    async def resume() -> None:
        before = len(client.calls)
        stats = await run_synthesis(
            EchoPipeline(),
            _args(4),
            client,
            output,
            concurrency=2,
            batch_size=2,
        )
        assert stats.skipped == 3
        assert stats.selected == 1
        assert stats.pending == 1
        assert client.calls[before:] == ["job-2"]

    asyncio.run(resume())

    async def retry_errors() -> None:
        client.replies["job-1"] = "fixed"
        client.replies["job-2"] = "fixed too"
        stats = await run_synthesis(
            EchoPipeline(),
            _args(4),
            client,
            output,
            concurrency=2,
            batch_size=2,
            retry_errors=True,
        )
        assert stats.wrote == 2
        assert stats.errors == 0
        assert stats.pending == 0

    asyncio.run(retry_errors())
    assert {row["id"] for row in _read(output / "jobs.jsonl")} == {"job-0", "job-1", "job-2", "job-3"}
    assert not (output / "errors.jsonl").exists()
    train = _read(output / "train.jsonl")
    assert len(train) == 4


def test_limit_then_continue(tmp_path: Path):
    client = FakeClient()
    output = tmp_path / "run"

    async def scenario() -> None:
        first = await run_synthesis(
            EchoPipeline(),
            _args(3),
            client,
            output,
            concurrency=1,
            batch_size=1,
            limit=1,
        )
        assert first.wrote == 1
        second = await run_synthesis(
            EchoPipeline(),
            _args(3),
            client,
            output,
            concurrency=1,
            batch_size=8,
        )
        assert second.skipped == 1
        assert second.wrote == 2

    asyncio.run(scenario())
    assert len(_read(output / "train.jsonl")) == 3


def test_duplicate_job_ids_fail(tmp_path: Path):
    class Dupes(EchoPipeline):
        def jobs(self, args):
            del args
            yield Job(id="same", input={})
            yield Job(id="same", input={})

    async def scenario() -> None:
        with pytest.raises(ValueError, match="duplicate"):
            await run_synthesis(Dupes(), _args(0), FakeClient(), tmp_path / "run")

    asyncio.run(scenario())


def test_resume_ignores_a_torn_trailing_line(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    torn = (
        json.dumps(
            {
                "id": "job-0",
                "pipeline": "echo",
                "examples": [
                    {
                        "id": "job-0#0",
                        "messages": [
                            {"role": "user", "content": "job-0"},
                            {"role": "assistant", "content": "kept"},
                        ],
                        "meta": {},
                    }
                ],
                "raw": "kept",
            }
        )
        + "\n"
        + '{"id": "job-1", "raw": "cut'
    )
    (output / "jobs.jsonl").write_bytes(torn.encode())
    client = FakeClient()

    async def scenario() -> None:
        stats = await run_synthesis(
            EchoPipeline(),
            _args(2),
            client,
            output,
            concurrency=1,
            batch_size=1,
        )
        assert stats.skipped == 1
        assert stats.wrote == 1
        assert client.calls == ["job-1"]

    asyncio.run(scenario())
    ids = [row["id"] for row in _read(output / "jobs.jsonl")]
    assert ids == ["job-0", "job-1"]


class TwoStepPipeline(Pipeline):
    name = "two_step"
    description = "Draft, then revise."

    def jobs(self, args):
        del args
        yield Job(id="only", input={"text": "Hello"})

    async def produce(self, job, args, client):
        del args
        draft, revised = await asyncio.gather(
            client.complete(ChatRequest(messages=[{"role": "user", "content": "draft"}])),
            client.complete(ChatRequest(messages=[{"role": "user", "content": "revise"}])),
        )
        return Produced(
            examples=[
                Example(
                    messages=[
                        {"role": "user", "content": job.input["text"], "trainable": False},
                        {"role": "assistant", "content": revised, "trainable": True},
                    ]
                )
            ],
            raw=[draft, revised],
        )


def test_multi_step_job_limits_calls_and_resumes(tmp_path: Path):
    client = FakeClient()
    output = tmp_path / "run"

    async def scenario() -> None:
        dry = await run_synthesis(
            TwoStepPipeline(),
            _args(0),
            client,
            output,
            concurrency=1,
            batch_size=1,
            dry_run=True,
        )
        assert dry.selected == 1
        assert client.calls == []
        assert not output.exists()
        stats = await run_synthesis(
            TwoStepPipeline(),
            _args(0),
            client,
            output,
            concurrency=1,
            batch_size=1,
        )
        assert stats.wrote == 1
        assert client.max_inflight == 1
        assert sorted(client.calls) == ["draft", "revise"]
        again = await run_synthesis(
            TwoStepPipeline(),
            _args(0),
            client,
            output,
            concurrency=1,
            batch_size=1,
        )
        assert again.selected == 0

    asyncio.run(scenario())
    job = _read(output / "jobs.jsonl")[0]
    assert job["raw"] == ["answer draft", "answer revise"]
    row = _read(output / "train.jsonl")[0]
    assert row["messages"][1]["trainable"] is True
    assert row["messages"][0]["trainable"] is False


def test_limit_must_be_positive(tmp_path: Path):
    async def scenario() -> None:
        with pytest.raises(ValueError, match="limit"):
            await run_synthesis(EchoPipeline(), _args(1), FakeClient(), tmp_path / "run", limit=0)

    asyncio.run(scenario())


def test_dry_run_writes_nothing(tmp_path: Path):
    client = FakeClient()
    output = tmp_path / "run"

    async def scenario() -> None:
        stats = await run_synthesis(
            EchoPipeline(),
            _args(2),
            client,
            output,
            dry_run=True,
        )
        assert stats.selected == 2
        assert client.calls == []
        assert not output.exists()

    asyncio.run(scenario())
