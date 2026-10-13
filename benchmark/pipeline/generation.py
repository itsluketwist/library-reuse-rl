"""Shared LLM client for the LibReuseBench v2 pipeline.

`complete_json` is the one Anthropic client call every stage uses; each stage's own prompt and
checks live with its script (the task generation in 02_generate.py, the agent review in
03_agent_review.py). Calls go through the Anthropic SDK directly, and every call's token use is
added to a running total, so a run can print what it spent (see usage_summary).
"""

import json
import threading
from typing import Any

import anthropic


# the model every stage uses, and how hard it thinks. opus 5.5 always thinks, and its own default
# effort is medium, so the effort is always set explicitly
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"

# the most tokens one reply may use. thinking counts towards it, and a generation reply holds two
# full solutions, one of which may run to hundreds of lines
MAX_TOKENS = 64000

# claude-opus-5-5's price in US dollars per million tokens, for the cost estimate usage_summary
# prints. the estimate is only right for that model
PRICE_PER_MILLION = {"input_tokens": 4.00, "output_tokens": 20.00}

# one shared client - reads its credentials from the environment and retries 429/5xx itself
_client = anthropic.Anthropic()

# the tokens used by every call this process has made. stages call the model from several
# threads at once, so the totals are only changed while holding the lock
_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
_usage_lock = threading.Lock()


def complete_json(
    prompt: str,
    schema: dict[str, Any],
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    effort: str | None = DEFAULT_EFFORT,
) -> tuple[dict[str, Any], str]:
    """Prompt the model with a structured-output schema and return the parsed object + model.

    Uses the API's structured outputs (`output_config.format`), so the reply is guaranteed to
    match `schema` - no fenced-code or prose parsing needed. `effort` ("low" to "max") sets how
    much the model thinks; left as None, the model's own default applies. The returned model id
    is the exact resolved snapshot the API served (not the alias requested), for provenance.

    Returns a (parsed object, resolved model id) tuple; the object is empty if it could not be
    read (e.g. a refusal, or a reply cut off at max_tokens).
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

    with _usage_lock:
        _usage["calls"] += 1
        _usage["input_tokens"] += message.usage.input_tokens
        _usage["output_tokens"] += message.usage.output_tokens

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


def usage_summary() -> str:
    """Describe the tokens every call so far has used, with their cost at PRICE_PER_MILLION.

    Returns a one-line summary to print at the end of a run.
    """
    with _usage_lock:
        usage = dict(_usage)
    cost = sum(
        usage[kind] * price / 1_000_000 for kind, price in PRICE_PER_MILLION.items()
    )
    return (
        f"{usage['calls']} model call(s): {usage['input_tokens']:,} input tokens, "
        f"{usage['output_tokens']:,} output tokens, about ${cost:.2f} at "
        f"{DEFAULT_MODEL} prices"
    )
