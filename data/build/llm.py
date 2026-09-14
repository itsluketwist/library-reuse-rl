"""Shared LLM client for the data-build pipeline (see data/README.md).

`complete` / `complete_json` mirror benchmark/pipeline/generation.py's client - a separate copy,
not imported, so data/ and benchmark/ remain independent packages. Used only by the offline
cache-regeneration CLI (data/build/library/augment.py); build.py itself never calls an LLM and
needs no API key.
"""

import json
import os
from typing import Any

import anthropic


# default generation model for the data-build pipeline's LLM-generated prompt variants,
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
