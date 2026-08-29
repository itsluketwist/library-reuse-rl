"""Generate responses from a model for evaluation, using vLLM for batched offline inference."""

from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .stop_reason import map_stop_reason


def generate_responses(
    llm: LLM,
    prompts: list[list[dict]],
    decoding_settings: dict[str, Any],
    lora_request: LoRARequest | None = None,
) -> list[tuple[str, str]]:
    """
    Generate one response per prompt from `llm`.

    `prompts` is a list of chat-messages lists (see load_eval_dataset).
    `decoding_settings` is a decoding profile dict (see eval_decoding_profiles in
    config/training.yaml); temperature 0 is vllm's greedy-decoding convention.
    `lora_request`, when given, applies a trained adapter on top of the base model.

    vllm handles chat templating, batching, and scheduling internally, and returns
    outputs in the same order as `prompts`.

    `enable_thinking=False` matches the non-reasoning baseline the api-served models in
    config/models.yaml are configured for (DeepSeek's "thinking" disabled, GPT's
    "reasoning_effort" set to "none"): local Qwen3 models default their chat template to
    thinking mode on, which otherwise burns the whole max_new_tokens budget on a reasoning
    trace before ever emitting an answer, truncating every response with no code to score.
    Templates that don't define the variable ignore it, so this is a no-op for other models.

    Returns a list of (response, stop_reason) tuples.
    """
    sampling_params = SamplingParams(
        temperature=decoding_settings["temperature"],
        top_p=decoding_settings["top_p"],
        top_k=decoding_settings.get("top_k", 0),
        max_tokens=decoding_settings["max_new_tokens"],
    )

    outputs = llm.chat(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
        chat_template_kwargs={"enable_thinking": True},
    )

    return [
        (output.outputs[0].text, map_stop_reason(output.outputs[0].finish_reason))
        for output in outputs
    ]
