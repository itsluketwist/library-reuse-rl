"""Generate responses from a hosted chat model over an OpenAI-compatible API.

Used for models evaluated by API instead of local weights (see the "api" provider in
config/models.yaml) - DeepSeek and OpenAI's own models both speak this same chat
completions protocol, so one client implementation covers both, distinguished only by
`base_url` and which API key env var holds the credential.
"""

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .stop_reason import map_stop_reason


# requests in flight at once; eval datasets here are small (tens to low hundreds of rows)
# so this is a throughput/rate-limit tradeoff, not a batching concern like vllm's
DEFAULT_MAX_WORKERS = 8


def _build_client(
    base_url: str | None,
    api_key_env: str,
) -> OpenAI:
    """
    Build an OpenAI-SDK client for the given `base_url` (None keeps the SDK's default,
    OpenAI's own API), authenticated from the `api_key_env` environment variable (loaded
    from .env by scripts/setup_job.sh).

    Returns the client.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable '{api_key_env}' is not set - it must hold the API key "
            "for this model (see api_key_env in config/models.yaml). Add it to .env."
        )
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=5)


# default request field for the generation-length cap. newer OpenAI reasoning-family
# models (e.g. gpt-5.6-luna) reject this legacy name and require "max_completion_tokens"
# instead (see "max_tokens_param" in config/models.yaml); DeepSeek's endpoint accepts the
# legacy name fine, so this stays the default for every entry that doesn't override it.
DEFAULT_MAX_TOKENS_PARAM = "max_tokens"


def _complete_one(
    client: OpenAI,
    model: str,
    prompt: list[dict],
    decoding_settings: dict[str, Any],
    extra_params: dict[str, Any] | None,
    max_tokens_param: str,
) -> tuple[str, str]:
    """
    Request a single chat completion for `prompt` (a chat-messages list, see
    load_eval_dataset). `decoding_settings` is a decoding profile dict (see
    eval_decoding_profiles in config/training.yaml); its top_k has no API equivalent and
    is ignored. `max_tokens_param` names the request field the generation-length cap is
    sent under (see DEFAULT_MAX_TOKENS_PARAM). `extra_params` (see "extra_params" in
    config/models.yaml), when given, is sent via the SDK's `extra_body` so provider-
    specific fields outside the standard OpenAI schema (e.g. DeepSeek's "thinking") reach
    the request.

    Returns a (response, stop_reason) tuple.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=cast(Iterable[ChatCompletionMessageParam], prompt),
        temperature=decoding_settings["temperature"],
        top_p=decoding_settings["top_p"],
        extra_body=extra_params,
        **{max_tokens_param: decoding_settings["max_new_tokens"]},
    )
    choice = completion.choices[0]
    return choice.message.content or "", map_stop_reason(choice.finish_reason)


def generate_api_responses(
    model: str,
    base_url: str | None,
    api_key_env: str,
    prompts: list[list[dict]],
    decoding_settings: dict[str, Any],
    extra_params: dict[str, Any] | None = None,
    max_tokens_param: str = DEFAULT_MAX_TOKENS_PARAM,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[list[tuple[str, str]]]:
    """
    Generate one or more sampled responses per prompt from the hosted `model`, via an
    OpenAI-compatible chat completions API.

    `prompts` is a list of chat-messages lists (see load_eval_dataset).
    `decoding_settings["n_samples"]` (default 1) sets how many independent completions
    are requested per prompt - matching generate_responses' vllm-side n_samples (e.g. for
    LibReuseBench's pass@k or library-usage's per-response reuse statistics, see
    src/eval/scoring.py's MULTI_SAMPLE_EVALUATORS). Not every OpenAI-compatible provider
    supports the request-level `n` parameter, so each sample is its own independent
    request rather than relying on it - every (prompt, sample) pair is requested
    concurrently (`max_workers` at a time), since each is an independent network call
    rather than a batched local forward pass. The SDK client retries transient errors
    (rate limits, 5xx) itself. `extra_params` and `max_tokens_param` are forwarded to
    every request (see _complete_one) - use them for provider-specific fields/quirks the
    standard decoding profile doesn't cover.

    Returns a list aligned with `prompts`, each element a list of that prompt's
    (response, stop_reason) samples (length `n_samples`).
    """
    client = _build_client(base_url=base_url, api_key_env=api_key_env)
    n_samples = decoding_settings.get("n_samples", 1)
    # repeat each prompt n_samples times, so every sample is its own independent request
    requests = [prompt for prompt in prompts for _ in range(n_samples)]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        flat = list(
            pool.map(
                lambda prompt: _complete_one(
                    client,
                    model,
                    prompt,
                    decoding_settings,
                    extra_params,
                    max_tokens_param,
                ),
                requests,
            )
        )

    return [flat[i * n_samples : (i + 1) * n_samples] for i in range(len(prompts))]
