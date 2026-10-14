"""Synthesize chat-format training data by calling a chat completions API.

A template reads source records, builds a prompt, and turns the reply into
SFT conversations. The runner executes those jobs with concurrency, batched
scheduling, and resume. See synthesis/README.md.
"""

from synthesis.client import ChatClient, build_chat_body
from synthesis.jsonutil import parse_json_object
from synthesis.messages import chat_messages, normalize_messages
from synthesis.pipeline import PIPELINES, Pipeline, get_pipeline, load_pipeline_file, register
from synthesis.runner import RunStats, run_synthesis
from synthesis.types import (
    ChatRequest,
    Example,
    FatalAPIError,
    Job,
    ParseError,
    Produced,
    RetryableError,
)

__all__ = [
    "PIPELINES",
    "ChatClient",
    "ChatRequest",
    "chat_messages",
    "Example",
    "FatalAPIError",
    "Job",
    "ParseError",
    "Pipeline",
    "Produced",
    "RetryableError",
    "RunStats",
    "build_chat_body",
    "get_pipeline",
    "load_pipeline_file",
    "normalize_messages",
    "parse_json_object",
    "register",
    "run_synthesis",
]
