# Synthetic Spec Class

Synthetic Spec Class synthesizes alignment training data from a model spec and a constitution, fine-tunes with [Tinker](https://tinker-docs.thinkingmachines.ai/), and evaluates the result.

Synthesis, Tinker supervised fine-tuning, and inference are in the repository. Evaluation against the spec is next.

## Layout

| Path | What it is |
|---|---|
| `synthesis/` | API synthesis runner and the pipelines that define each chain |
| `training/` | Tinker SFT and inference over the synthesized conversations |
| `model_spec/` | OpenAI Model Spec, imported with `git subtree` |
| `claude-constitution/` | Claude's constitution, imported with `git subtree` |
| `model_spec_evals/` | OpenAI's Model Spec eval harness, imported with `git subtree` |

The three upstream trees stay separate from this project's code. Their documents are CC0.

## Synthesis

A pipeline decides what to ask the model and how to read the answer. One job is usually one model call; `produce` is where a job makes several calls. The runner calls an OpenAI-compatible chat completions API, runs requests in batches with a concurrency limit, and resumes from JSONL checkpoints.

```bash
python -m synthesis list
python -m synthesis run section_conversations --help
```

Dry-run the Model Spec chain without calling an API:

```bash
python -m synthesis run section_conversations \
  --source model_spec/model_spec.md \
  --output runs/model-spec \
  --dry-run
```

Generate, then continue the same command after an interruption:

```bash
export OPENAI_API_KEY=...
python -m synthesis run section_conversations \
  --source model_spec/model_spec.md \
  --output runs/model-spec \
  --model gpt-4.1-mini \
  --concurrency 8 \
  --batch-size 32
```

`runs/model-spec/train.jsonl` is chat JSONL. Each line has `messages` plus `id`, `job_id`, and `meta`. Tinker Cookbook's `FromConversationFileBuilder` trains on the `messages` field.

Pipeline details, output files, and how to add a chain are in [synthesis/README.md](synthesis/README.md).

## Training and inference

SFT uses Tinker Cookbook. Inference samples with the same chat renderer and can read the newest sampler checkpoint from the training log.

```bash
pip install -e ".[train]"
export TINKER_API_KEY=...

python -m training sft \
  --data runs/model-spec \
  --log-dir logs/model-spec \
  --model Qwen/Qwen3-8B \
  --dry-run

python -m training infer \
  --log-dir logs/model-spec \
  --input runs/model-spec/train.jsonl \
  --output logs/model-spec/infer
```

Flags, resume behavior, and the prediction file are in [training/README.md](training/README.md).

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`python -m synthesis` also works from the repository root without installing.
