"""Generate responses from a model for evaluation, using vLLM for batched offline inference."""

from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def _stop_reason(finish_reason: str) -> str:
    """
    Map vllm's finish_reason to this project's stop_reason vocabulary: "stop" (natural
    end-of-turn) becomes "eos", "length" (hit max_tokens) passes through. Anything else
    passes through verbatim so it stays visible rather than inflating the overflow count.

    Returns the mapped stop reason.
    """
    return {"stop": "eos", "length": "length"}.get(finish_reason, finish_reason)


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
    )

    return [
        (output.outputs[0].text, _stop_reason(output.outputs[0].finish_reason))
        for output in outputs
    ]
