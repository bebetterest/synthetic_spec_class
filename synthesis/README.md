# Synthesis

The runner calls a chat completions API. A chain is a template with three functions: `read` yields one record per model call, `prompt` builds that call, and `parse` turns the reply into SFT conversations. The two defaults live in `synthesis/templates/`. A job is usually one model call. A class that overrides `produce` is for a job that makes several calls.

## Run

From the repository root:

```bash
python -m synthesis list
python -m synthesis run prompts \
  --input synthesis/examples/prompts.jsonl \
  --output runs/smoke \
  --model gpt-4.1-mini \
  --concurrency 4 \
  --batch-size 8
```

The same command resumes. Finished job ids in `jobs.jsonl`, and parse or fatal API failures in `errors.jsonl`, are skipped. A trailing partial line left by a crash is discarded and that job runs again.

| Flag | Meaning |
|---|---|
| `--concurrency` | Maximum requests in flight. Default 8. |
| `--batch-size` | How many pending jobs are scheduled together. Default 32. Use a value at least as large as `--concurrency`. The next batch starts only after the current one has been written. |
| `--limit` | Cap how many pending jobs this invocation runs. |
| `--retry-errors` | Run ids in `errors.jsonl` again. Successful jobs are still skipped. |
| `--overwrite` | Delete this output directory's checkpoints, then start over. |
| `--dry-run` | Print the first pending request. No API call and no files written. |
| `--max-retries` | Extra attempts for HTTP 408, 409, 429, and 5xx, and for network errors. Default 5. |
| `--base-url` | API root. `https://api.openai.com/v1` or a full `.../chat/completions` URL. Environment: `OPENAI_BASE_URL`. |
| `--api-key` | Bearer token. Environment: `OPENAI_API_KEY`. Omit for a local server. |
| `--model` | Model name. Environment: `SYNTHESIS_MODEL`. |

A retryable failure is not marked done. Run the same command to try it again. A parse failure or a non-retryable HTTP error is marked done so a bad row does not spin forever. Fix the pipeline or the request, then pass `--retry-errors`.

Exit status is `1` when any selected job is still pending, and `0` when every selected job was written or recorded as an error. Ctrl-C exits `130`. Completed jobs from that batch are kept.

## Output

```text
runs/<name>/
  jobs.jsonl      one line per completed API call; this is the resume checkpoint
  train.jsonl     one line per training conversation
  errors.jsonl    parse failures and fatal API errors; skipped on resume
  failures.jsonl  retryable failures that will be tried again; safe to delete
```

A training line:

```json
{
  "id": "model_spec.md/chain_of_command#0",
  "job_id": "model_spec.md/chain_of_command",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {"pipeline": "section_conversations", "section_id": "chain_of_command"}
}
```

`train.jsonl` is rebuilt from `jobs.jsonl` at the start and end of a run. `python -m synthesis materialize runs/<name>` rebuilds it on its own.

Tinker SFT reads `messages`. `system`, `developer`, `user`, `assistant`, and `tool` are accepted. Every saved conversation must include a user message and an assistant message.

## Templates

`list` shows the registered chains. A built-in name runs as before:

```bash
python -m synthesis run section_conversations --help
```

A new chain is a copy of a file in `synthesis/templates/`. Edit `read`, `prompt`, and `parse`. `add_arguments` adds flags that belong to that file. Set `name` to a lower-case identifier when the filename is not already one.

```bash
python -m synthesis run --template path/to/copy.py --output runs/name --source policy.md
```

`read` yields a dict with a string `id`, or a `Job`. `prompt` returns a `ChatRequest`, a user-message string, or a list of messages. `parse` returns `Example`s, a list of message lists, or one message list. Each saved conversation still needs a user message and an assistant message.

`prompt` and `parse` receive the same dict `read` yielded, including `id`.

### `section_conversations`

Split a Markdown spec or constitution into sections and ask the model for conversations that follow each section. The prompt is the `PROMPT` string at the top of `synthesis/templates/section_conversations.py`.

```bash
python -m synthesis run section_conversations \
  --source model_spec/model_spec.md \
  --heading-level 2 \
  --examples-per-section 4 \
  --output runs/model-spec \
  --model gpt-4.1-mini

python -m synthesis run section_conversations \
  --source claude-constitution/20260120-constitution.md \
  --output runs/constitution \
  --model gpt-4.1-mini
```

Heading level 2 matches both documents: level 1 is a part, level 2 is a principle or topic, and deeper headings stay inside the section body. Headings inside fenced code are ignored. Section ids prefer an explicit `{#anchor}`. The job id is `<filename>/<section id>`, so use a separate `--output` directory per source file.

`--max-section-chars` (default 12000) truncates the text placed in the prompt. `--no-json-mode` drops `response_format=json_object` for servers that reject it.

The generated conversations are demonstrations of the behavior. They are not transcripts about the policy document. That instruction lives in `PROMPT` and is meant to be edited.

### `prompts`

Send a JSONL file of prefixes. Each row needs `id` and either `messages` or `prompt`. Optional `temperature`, `model`, and `meta` are per row. `--system` prepends a system message to every request and stores it in the training example.

```json
{"id": "smoke-1", "prompt": "Say hello in one short sentence."}
{"id": "smoke-2", "messages": [{"role": "user", "content": "Name one primary color."}], "temperature": 0}
```

Use this when each row is already a chat prefix. A row from an earlier `train.jsonl` works after you drop the final assistant message, or leave it when the model should continue from that reply. A boolean `trainable` on an input message is stored on the training row and left out of the API request.

## Add a chain

Copy `synthesis/templates/section_conversations.py` or `synthesis/templates/prompts.py`. The three functions are the whole chain. `--template` may be repeated. When one file is passed and no chain name is given, that file is the chain.

```python
name = "gloss"
description = "Turn each line into one SFT conversation."

def add_arguments(parser):
    parser.add_argument("--input", required=True)

def read(args):
    for index, line in enumerate(open(args.input, encoding="utf-8"), start=1):
        text = line.strip()
        if text:
            yield {"id": f"line-{index}", "text": text}

def prompt(item, args):
    del args
    return f"Translate to French:\n{item['text']}"

def parse(item, content, args):
    del args
    if not content.strip():
        raise ValueError("empty completion")
    return [
        {"role": "user", "content": item["text"]},
        {"role": "assistant", "content": content.strip()},
    ]
```

```bash
python -m synthesis run --template chains/gloss.py --output runs/gloss --input lines.txt
```

`name` is a lower-case identifier and becomes the CLI command when you pass it explicitly. Do not reuse runner flags (`--output`, `--model`, `--concurrency`, and the rest listed on `Pipeline`).

Job ids must be stable across runs. The runner refuses duplicate ids. `parse` may return several conversations from one reply; they are stored as `<job id>#0`, `<job id>#1`, and so on.

`parse_json_object` in `synthesis.jsonutil` accepts a JSON object, a fenced object, or an object surrounded by prose. Raise `ParseError` from `parse` when the reply cannot be used. The runner records that and skips the job until `--retry-errors`.

A `Pipeline` subclass loaded with `--pipeline-file` is for a job that calls the model more than once. A module dropped in `synthesis/pipelines/` is imported on startup.

### Several model calls in one job

Override `produce` and call `client.complete` once per step. The job is saved only after `produce` returns, so a later step that fails runs the whole job again. `--concurrency` limits `complete` calls across jobs. Return a list of `Example`, or a `Produced` when you want the step outputs kept in `jobs.jsonl` as `raw`.

```python
async def produce(self, job, args, client):
    draft = await client.complete(ChatRequest(messages=[
        {"role": "user", "content": f"Draft a reply:\n{job.input['text']}"},
    ]))
    final = await client.complete(ChatRequest(messages=[
        {"role": "user", "content": f"Revise this reply:\n{draft}"},
    ]))
    return self.parse(job, final, args)
```

`preview` controls the dry-run printout. The default prints `request()`. Return `None` when the first call is built inside `produce`.

A later stage can also be its own command. Its `jobs()` reads the previous `train.jsonl` with `read_jsonl` from `synthesis.io`. Use a new `--output` directory so the two checkpoints stay separate. `prompts` is that pattern when each row is already a chat prefix.
