"""Generate responses from a hosted chat model over an OpenAI-compatible API.

Used for models evaluated by API rather than local weights (the "api" provider in
config/models.yaml). DeepSeek, OpenAI, Zhipu and Moonshot all speak the same chat
completions protocol, so one client covers them all, differing only by base_url and
which environment variable holds the key.
"""

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from src.eval.stop_reason import map_stop_reason
from src.utils.log import log


# requests in flight at once; eval datasets are small (tens to low hundreds of rows), so this
# is a rate-limit tradeoff rather than a batching concern like vllm's
DEFAULT_MAX_WORKERS = 8

# newer openai reasoning models reject the legacy "max_tokens" and need this name instead
DEFAULT_MAX_TOKENS_PARAM = "max_tokens"

# stop reason recorded for a request that never succeeded (see generate_api_responses)
FAILED_STOP_REASON = "error"


class ApiSettings(NamedTuple):
    """How to reach a hosted model's OpenAI-compatible endpoint (see config/models.yaml).

    `base_url` of None keeps the SDK default (OpenAI's own API). `extra_params` is sent
    verbatim for provider-specific fields outside the standard schema (e.g. DeepSeek's
    "thinking"), and `max_tokens_param` names the field the length cap is sent under.
    """

    api_key_env: str
    base_url: str | None = None
    extra_params: dict[str, Any] | None = None
    max_tokens_param: str = DEFAULT_MAX_TOKENS_PARAM


def _build_client(
    api: ApiSettings,
) -> OpenAI:
    """
    Build an OpenAI-SDK client, authenticated from the environment variable `api` names
    (loaded from .env by scripts/setup_job.sh).

    Returns the client.
    """
    api_key = os.environ.get(api.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable '{api.api_key_env}' is not set - it must hold the API "
            "key for this model (see api_key_env in config/models.yaml). Add it to .env."
        )
    return OpenAI(api_key=api_key, base_url=api.base_url, max_retries=5)


def _complete_one(
    client: OpenAI,
    model: str,
    prompt: list[dict],
    decoding_settings: dict[str, Any],
    api: ApiSettings,
) -> tuple[str, str]:
    """
    Request a single chat completion for `prompt` (a chat-messages list, see
    load_eval_dataset). `decoding_settings`' top_k has no API equivalent and is ignored.

    Returns a (response, stop_reason) tuple.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=cast(Iterable[ChatCompletionMessageParam], prompt),
            temperature=decoding_settings["temperature"],
            top_p=decoding_settings["top_p"],
            extra_body=api.extra_params,
            **{api.max_tokens_param: decoding_settings["max_new_tokens"]},
        )
    except Exception as exc:
        # the SDK already retried; failing this one sample beats discarding every other
        # (paid) generation in the run by raising out of the thread pool
        return f"API request failed: {exc}", FAILED_STOP_REASON

    choice = completion.choices[0]
    text = choice.message.content or ""

    # reasoning models return their thinking in a separate field - fold it back into the
    # text as a <think> block, so strip_reasoning and the n_reasoning metric see the same
    # shape they do from vllm (see src/response/parser.py)
    reasoning = getattr(choice.message, "reasoning_content", None)
    if reasoning:
        text = f"<think>{reasoning}</think>{text}"

    return text, map_stop_reason(choice.finish_reason)


def generate_api_responses(
    model: str,
    api: ApiSettings,
    prompts: list[list[dict]],
    decoding_settings: dict[str, Any],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[list[tuple[str, str]]]:
    """
    Generate one or more sampled responses per prompt from the hosted `model`.

    `prompts` is a list of chat-messages lists (see load_eval_dataset).
    `decoding_settings["n_samples"]` (default 1) sets how many completions are requested
    per prompt, matching generate_vllm_responses' vllm-side n_samples. Not every provider
    supports the request-level `n` parameter, so each sample is its own request, and all
    of them run concurrently (`max_workers` at a time) since each is an independent
    network call rather than a batched local forward pass.

    Returns a list aligned with `prompts`, each element a list of that prompt's
    (response, stop_reason) samples (length `n_samples`).
    """
    client = _build_client(api)
    n_samples = decoding_settings.get("n_samples", 1)
    # repeat each prompt n_samples times, so every sample is its own independent request
    requests = [prompt for prompt in prompts for _ in range(n_samples)]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        flat = list(
            tqdm(
                pool.map(
                    lambda prompt: _complete_one(
                        client,
                        model,
                        prompt,
                        decoding_settings,
                        api,
                    ),
                    requests,
                ),
                total=len(requests),
                desc="api generation",
            )
        )

    failed = sum(1 for _text, stop_reason in flat if stop_reason == FAILED_STOP_REASON)
    if failed:
        log(f"WARNING: {failed}/{len(flat)} API requests failed - see the responses")

    return [flat[i * n_samples : (i + 1) * n_samples] for i in range(len(prompts))]
