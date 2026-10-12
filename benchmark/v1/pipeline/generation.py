"""Shared LLM client for the LibReuseBench pipeline.

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
    effort: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Prompt the model with a structured-output schema and return the parsed object + model.

    Uses the API's structured outputs (`output_config.format`), so the reply is guaranteed to
    match `schema` - no fenced-code or prose parsing needed. `effort` ("low" to "max") sets how
    much the model thinks; left as None, the model's own default applies. The returned model id
    is the exact resolved snapshot the API served (not the alias requested), for provenance.

    Returns a (parsed object, resolved model id) tuple; the object is empty if it could not be
    read (e.g. a refusal).
    """
    output_config: dict[str, Any] = {
        "format": {"type": "json_schema", "schema": schema}
    }
    if effort is not None:
        output_config["effort"] = effort
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": output_config,
    }
    if system is not None:
        kwargs["system"] = system

    # streamed, then gathered into one message: a reply with a large max_tokens (thinking
    # included) can take longer than a single non-streaming request is allowed to
    with _client.messages.stream(**kwargs) as stream:
        message = stream.get_final_message()
    text = "".join(block.text for block in message.content if block.type == "text")
    try:
        # strict=False lets a raw control character (e.g. a tab inside generated code) stand
        # inside a json string - the default rejects the whole reply
        parsed = json.loads(text, strict=False)
        data = parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as error:
        # say why, so an unreadable reply can be told apart from a refusal or a truncation
        print(
            f"unreadable reply: stop_reason={message.stop_reason}, "
            f"{len(text)} characters, {error}"
        )
        data = {}
    return data, message.model
