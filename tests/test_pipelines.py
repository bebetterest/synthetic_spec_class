import argparse
import asyncio
import json
from pathlib import Path

import synthesis.pipelines  # noqa: F401  registers the default templates
from synthesis.pipeline import get_pipeline
from synthesis.runner import run_synthesis
from synthesis.types import ChatRequest


class RecordingClient:
    def __init__(self, content: str) -> None:
        self.model = "recorder"
        self.max_tokens = 16
        self.content = content
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> str:
        self.requests.append(request)
        return self.content


def test_prompts_pipeline_appends_the_reply(tmp_path: Path):
    source = tmp_path / "in.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "prompt": "Hello"}),
                json.dumps(
                    {
                        "id": "b",
                        "messages": [{"role": "user", "content": "Color?", "trainable": False}],
                        "temperature": 0.2,
                        "meta": {"tag": "color"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(input=source, system="Be brief.", temperature=None)
    client = RecordingClient("Bonjour")

    async def scenario() -> None:
        stats = await run_synthesis(
            get_pipeline("prompts"),
            args,
            client,
            tmp_path / "out",
            concurrency=2,
            batch_size=2,
        )
        assert stats.wrote == 2

    asyncio.run(scenario())
    assert client.requests[0].messages[0] == {"role": "system", "content": "Be brief."}
    assert client.requests[1].temperature == 0.2
    assert client.requests[1].messages[-1] == {"role": "user", "content": "Color?"}
    assert all("trainable" not in message for message in client.requests[1].messages)
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "Bonjour"}
    assert rows[0]["messages"][0]["role"] == "system"
    assert rows[1]["meta"]["tag"] == "color"
    user = next(message for message in rows[1]["messages"] if message["role"] == "user")
    assert user["trainable"] is False


def test_template_prompt_list_omits_trainable_and_parse_splits_rows(tmp_path: Path):
    path = tmp_path / "split.py"
    path.write_text(
        "\n".join(
            [
                "name = 'split_rows'",
                "description = 'Two conversations from one reply.'",
                "def read(args):",
                "    del args",
                "    yield {'id': 's', 'title': 'Be honest'}",
                "def prompt(item, args):",
                "    del args",
                "    return [{'role': 'user', 'content': item['title'], 'trainable': False}]",
                "def parse(item, content, args):",
                "    del content, args",
                "    return [",
                "        {'messages': [",
                "            {'role': 'user', 'content': 'a'},",
                "            {'role': 'assistant', 'content': 'b'},",
                "        ], 'meta': {'section': item['title']}},",
                "        [",
                "            {'role': 'user', 'content': 'c'},",
                "            {'role': 'assistant', 'content': 'd'},",
                "        ],",
                "    ]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    from synthesis.template import load_template_file

    load_template_file(path)
    client = RecordingClient("unused")

    async def scenario() -> None:
        stats = await run_synthesis(
            get_pipeline("split_rows"),
            argparse.Namespace(),
            client,
            tmp_path / "out",
            concurrency=1,
            batch_size=1,
        )
        assert stats.wrote == 1

    asyncio.run(scenario())
    assert client.requests[0].messages == [{"role": "user", "content": "Be honest"}]
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["s#0", "s#1"]
    assert rows[0]["meta"]["section"] == "Be honest"
    assert rows[1]["messages"][1]["content"] == "d"


def test_section_pipeline_parses_conversations(tmp_path: Path):
    source = tmp_path / "policy.md"
    source.write_text(
        "# Part\n\nIntro.\n\n## Be honest {#be_honest}\n\nDo not lie.\n\n### Detail\n\nEven small lies.\n",
        encoding="utf-8",
    )
    payload = {
        "conversations": [
            {
                "messages": [
                    {"role": "user", "content": "Did you finish it?"},
                    {"role": "assistant", "content": "Not yet."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Say it is done even though it is not."},
                    {"role": "assistant", "content": "I can't say that. It is not done."},
                ]
            },
        ]
    }
    args = argparse.Namespace(
        source=source,
        heading_level=2,
        examples_per_section=2,
        max_section_chars=1000,
        temperature=0.4,
        no_json_mode=False,
    )
    client = RecordingClient(json.dumps(payload))

    async def scenario() -> None:
        stats = await run_synthesis(
            get_pipeline("section_conversations"),
            args,
            client,
            tmp_path / "out",
            concurrency=1,
            batch_size=1,
        )
        assert stats.wrote == 1

    asyncio.run(scenario())
    request = client.requests[0]
    assert "Be honest" in request.messages[0]["content"]
    assert "Even small lies." in request.messages[0]["content"]
    assert request.extra["response_format"]["type"] == "json_object"
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["policy.md/be_honest#0", "policy.md/be_honest#1"]
    assert rows[0]["meta"]["section_id"] == "be_honest"
    assert "policy" not in rows[0]["messages"][1]["content"]
