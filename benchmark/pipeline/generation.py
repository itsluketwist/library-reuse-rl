"""Shared LLM client for the BigCodeBenchReuse pipeline.

`complete` / `complete_json` are the shared Anthropic client used by every stage; each stage's own
prompt/parse logic lives with its script (the judge in 02_filter.py, the rewrite in 03_rewrite.py,
the gold + test generation in 04, the library solution in 05). Calls go through the Anthropic SDK
directly (Claude by default, overridable via EVAL_DATASET_MODEL).
"""

import json
import os
from typing import Any

import anthropic


# default generation model for the code-writing stages (rewrite, solution generation),
# overridable so a scale run can trade quality for cost without code changes
DEFAULT_GEN_MODEL = os.environ.get("EVAL_DATASET_MODEL", "claude-sonnet-5")

# one shared client - reads ANTHROPIC_API_KEY from the environment and retries 429/5xx itself
_client = anthropic.Anthropic()


def complete(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_GEN_MODEL,
    max_tokens: int = 4096,
) -> str:
    """Prompt the generation model once and return its text reply.

    No sampling parameters are set: temperature/top_p are rejected by the current Claude models,
    and omitting `thinking` keeps behaviour consistent across model choices.

    Returns the model's completion text (concatenated text blocks).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system is not None:
        kwargs["system"] = system

    message = _client.messages.create(**kwargs)
    return "".join(block.text for block in message.content if block.type == "text")


def complete_json(
    prompt: str,
    schema: dict[str, Any],
    system: str | None = None,
    model: str = DEFAULT_GEN_MODEL,
    max_tokens: int = 4096,
) -> tuple[dict[str, Any], str]:
    """Prompt the model with a structured-output schema and return the parsed object + model.

    Uses the API's structured outputs (`output_config.format`), so the reply is guaranteed to
    match `schema` - no fenced-code or prose parsing needed. The returned model id is the exact
    resolved snapshot the API served (not the alias requested), for provenance.

    Returns a (parsed object, resolved model id) tuple; the object is empty if it could not be
    read (e.g. a refusal).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    if system is not None:
        kwargs["system"] = system

    message = _client.messages.create(**kwargs)
    text = "".join(block.text for block in message.content if block.type == "text")
    try:
        parsed = json.loads(text)
        data = parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        data = {}
    return data, message.model
