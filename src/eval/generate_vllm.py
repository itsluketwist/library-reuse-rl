"""Generate responses from local model weights, using vllm for batched offline inference.

Used for models evaluated from their own weights (the default "vllm" provider in the
model registry), whether a released base model or one of a training run's epoch
checkpoints. The hosted-API counterpart is src/eval/generate_api.py.
"""

from typing import Any

from vllm import LLM, SamplingParams

from src.eval.reasoning_prefill import prefilled_reasoning_tag
from src.response.stop_reason import map_stop_reason


# how many of each prompt's final tokens to decode when looking for a prefilled opening
# reasoning tag - enough to cover the tag and its trailing whitespace
_PROMPT_END_TOKENS = 16


def generate_vllm_responses(
    llm: LLM,
    prompts: list[list[dict]],
    decoding_settings: dict[str, Any],
) -> list[list[tuple[str, str]]]:
    """
    Generate one or more sampled responses per prompt from `llm`.

    `prompts` is a list of chat-messages lists (see load_eval_dataset).
    `decoding_settings` is a decoding profile dict (see eval_decoding_profiles in
    config/training.yaml); temperature 0 is vllm's greedy-decoding convention.
    `decoding_settings["n_samples"]` (default 1) sets how many completions vllm samples
    per prompt - e.g. for LibReuseBench's pass@k or library-usage's per-response reuse
    statistics (see src/eval/scoring's MULTI_SAMPLE_EVALUATORS); its `enable_thinking`
    drives the chat template's reasoning mode.

    vllm handles chat templating, batching, and scheduling internally, and returns
    outputs in the same order as `prompts`.

    Returns a list aligned with `prompts`, each element a list of that prompt's
    (response, stop_reason) samples (length `n_samples`).
    """
    records = generate_vllm_records(llm, prompts, decoding_settings)
    return [
        [(sample["content"], sample["finish_reason"]) for sample in samples]
        for samples in records
    ]


def generate_vllm_records(
    llm: LLM,
    prompts: list[list[dict]],
    decoding_settings: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    """Generate responses with token counts for durable, independently scored output.

    Returns generation records grouped by prompt and sample.
    """
    sampling_params = SamplingParams(
        temperature=decoding_settings["temperature"],
        top_p=decoding_settings["top_p"],
        top_k=decoding_settings.get("top_k", 0),
        max_tokens=decoding_settings["max_new_tokens"],
        n=decoding_settings.get("n_samples", 1),
        seed=decoding_settings.get("seed"),
    )

    outputs = llm.chat(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
        chat_template_kwargs={
            "enable_thinking": decoding_settings.get("enable_thinking", True)
        },
    )

    # some chat templates end the prompt with the opening <think>, so the model's text starts
    # mid-reasoning. store the whole assistant turn by adding that tag back, so a response
    # cut off before </think> is scored as unfinished reasoning rather than as an answer
    tokenizer = llm.get_tokenizer()
    prefills = [
        prefilled_reasoning_tag(
            tokenizer.decode(list(output.prompt_token_ids or [])[-_PROMPT_END_TOKENS:]),
        )
        for output in outputs
    ]

    return [
        [
            {
                "content": prefill + sample.text,
                "finish_reason": map_stop_reason(sample.finish_reason),
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids or []),
                    "completion_tokens": len(sample.token_ids),
                },
            }
            for sample in output.outputs
        ]
        for output, prefill in zip(outputs, prefills)
    ]
