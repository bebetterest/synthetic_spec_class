import json
from pathlib import Path

from synthesis.cli import main


def test_list_includes_builtin_pipelines(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "section_conversations" in output
    assert "prompts" in output


def test_dry_run_prints_a_request_and_writes_nothing(tmp_path: Path, capsys):
    source = tmp_path / "policy.md"
    source.write_text("# Part\n\n## Rule {#rule}\n\nFollow the rule.\n", encoding="utf-8")
    output = tmp_path / "out"
    code = main(
        [
            "run",
            "section_conversations",
            "--source",
            str(source),
            "--output",
            str(output),
            "--dry-run",
            "--examples-per-section",
            "1",
        ]
    )
    assert code == 0
    assert not output.exists()
    captured = capsys.readouterr()
    assert "policy.md/rule" in captured.err
    assert "Follow the rule." in captured.err


def test_pipeline_file_is_loaded_before_the_command(tmp_path: Path, capsys):
    path = tmp_path / "file_chain.py"
    path.write_text(
        "\n".join(
            [
                "from synthesis.pipeline import Pipeline, register",
                "from synthesis.types import ChatRequest, Example, Job",
                "",
                "@register",
                "class FileChain(Pipeline):",
                "    name = 'file_chain'",
                "    description = 'Loaded from a file.'",
                "",
                "    def jobs(self, args):",
                "        del args",
                "        yield Job(id='only', input={})",
                "",
                "    def request(self, job, args):",
                "        del job, args",
                "        return ChatRequest(messages=[{'role': 'user', 'content': 'hello from file'}])",
                "",
                "    def parse(self, job, content, args):",
                "        del job, args",
                "        return [Example(messages=[{'role': 'user', 'content': 'hello from file'}, {'role': 'assistant', 'content': content}])]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["list", "--pipeline-file", str(path)]) == 0
    assert "file_chain" in capsys.readouterr().out
    output = tmp_path / "out"
    code = main(
        [
            "run",
            "--pipeline-file",
            str(path),
            "file_chain",
            "--output",
            str(output),
            "--dry-run",
        ]
    )
    assert code == 0
    assert "hello from file" in capsys.readouterr().err
    assert not output.exists()


def test_materialize_rebuilds_train_jsonl(tmp_path: Path, capsys):
    output = tmp_path / "out"
    output.mkdir()
    job = {
        "id": "job-1",
        "pipeline": "prompts",
        "examples": [
            {
                "id": "job-1#0",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
                "meta": {},
            }
        ],
        "raw": "Hello",
    }
    (output / "jobs.jsonl").write_text(json.dumps(job) + "\n", encoding="utf-8")
    assert main(["materialize", str(output)]) == 0
    train = (output / "train.jsonl").read_text(encoding="utf-8")
    assert json.loads(train)["job_id"] == "job-1"
    assert "training rows" in capsys.readouterr().out
